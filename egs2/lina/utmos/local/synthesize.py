import argparse
import logging
import os
import tarfile
import time
from pathlib import Path
from typing import Dict

import soundfile as sf
import yaml
from tqdm import tqdm

from espnet2.bin.tts_inference import Text2Speech
from espnet2.main_funcs.pack_funcs import unpack, find_path_and_change_it_recursive

cache_dir = "./.cache/"
device = "cpu"


# device = "cuda" # uncomment if you have a GPU and want to use it

def _safe_extract_tar(tar: tarfile.TarFile, path: Path):
    base = path.resolve()

    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(base)):
            raise Exception("Unsafe tar archive")

    tar.extractall(path)


def find_vocoder(search_dir: Path) -> str:
    for file in search_dir.rglob("*.pkl"):
        # print(f"found {str(file)}")
        return str(file)

    raise FileNotFoundError(f"No *.pkl file found in the {search_dir}")


def get_dict_from_cache(meta: Path) -> Dict[str, str]:
    if not meta.exists():
        raise FileNotFoundError(f"No meta.yaml found in the {meta}")

    outpath = meta.parent
    with meta.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
        yaml_files = d["yaml_files"]
        files = d["files"]

        for _, value in yaml_files.items():
            if not (outpath / value).exists():
                raise FileNotFoundError(f"No {outpath / value} found")
            with (outpath / value).open("r", encoding="utf-8") as f:
                d_int = yaml.safe_load(f)
            for path in outpath.rglob("*"):
                if path.is_file():
                    relative_path = path.relative_to(outpath)
                    d_int = find_path_and_change_it_recursive(d_int, str(relative_path), str(path))
            with (outpath / value).open("w", encoding="utf-8") as f:
                yaml.safe_dump(d_int, f)
        retval = {}
        for key, value in list(yaml_files.items()) + list(files.items()):
            if not (outpath / value).exists():
                raise FileNotFoundError(f"No {outpath / value} found")
            retval[key] = str(outpath / value)
        return retval


def extract_archive_cached(archive_path: str, cache_dir: str) -> Path:
    archive_path = Path(archive_path)
    if archive_path.is_dir():
        return archive_path  # already extracted

    cache_dir = Path(cache_dir)

    stat = archive_path.stat()

    # fast cache key (no hashing)
    key = f"{archive_path.name}_{stat.st_size}_{int(stat.st_mtime)}"

    extract_path = cache_dir / key
    marker = extract_path / ".done"

    if marker.exists():
        return extract_path

    extract_path.mkdir(parents=True, exist_ok=True)

    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path) as tar:
            _safe_extract_tar(tar, extract_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    marker.touch()
    return extract_path


def init_tts(am_file=None, vocoder_file=None):
    if not vocoder_file:
        vocoder_file = None  # make sure it's None, not empty string
    else:
        vocoder_dir = extract_archive_cached(archive_path=vocoder_file, cache_dir=cache_dir)
        vocoder_file = find_vocoder(vocoder_dir)
        print(f"vocoder_file: {vocoder_file}")

    print(f"\n===================================\nAM      = {am_file}")
    print(f"Vocoder = {vocoder_file if vocoder_file else 'Griffin-Lim'}")
    print(f"===================================")

    am_path = Path(am_file)
    if am_path.is_dir():
        meta = am_path / "meta.yaml"
        kwargs = get_dict_from_cache(meta=meta)
    else:
        kwargs = unpack(input_archive=am_file, outpath=cache_dir)
    print(kwargs)

    tts = Text2Speech.from_pretrained(
        vocoder_file=vocoder_file,
        device=device,
        **kwargs
    )

    def run_tts(text):
        started = time.perf_counter()
        res = tts(text)
        audio = res["wav"].cpu().numpy()
        generation_seconds = time.perf_counter() - started
        audio_duration = len(audio) / tts.fs
        print(f'  Duration: {audio_duration:.2f}s')
        print(f'  In: {generation_seconds:.2f}s')
        print(f'  RTF: {generation_seconds / audio_duration:.3f}x')
        return audio

    return run_tts, tts.fs


## workaround: fix parallel_wavegan for python3.12
import scipy.signal.windows
import sys

sys.modules['scipy.signal'].kaiser = scipy.signal.windows.kaiser


def main():
    parser = argparse.ArgumentParser(description="Synthesize audio",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--am", type=Path, help="AM model")
    parser.add_argument("--vocoder", type=Path, help="Vocoder model")
    parser.add_argument("--input", type=Path, help="Input file with text to synthesize")
    parser.add_argument("--output_dir", type=Path, help="Directory to save synthesized audio")
    args = parser.parse_args()

    logging.info(f"am {args.am}")
    logging.info(f"vocoder {args.vocoder}")
    logging.info(f"input {args.input}")
    logging.info(f"output_dir {args.output_dir}")

    tts, fs = init_tts(am_file=args.am, vocoder_file=args.vocoder)
    logging.info(f"READY: modelis įkeltas. FS: {fs}Hz\n")
    with open(args.input, "r") as f:
        lines = f.readlines()

    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Synthesizing"):
        line = line.strip()
        if not line:
            continue
        wav = tts(line)
        out_file = args.output_dir / f"output_{i}.wav"
        sf.write(out_file, wav, fs)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")

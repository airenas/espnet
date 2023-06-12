import argparse
import os
import sys
import time

import numpy as np
import torch
from parallel_wavegan.utils import load_model

from espnet2.bin.tts_inference import Text2Speech

fs = 22050


def to_int16(data):
    i = np.iinfo(np.int16)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (data * abs_max + offset).clip(i.min, i.max).astype(np.int16)


def write_wav(name, data):
    from scipy.io.wavfile import write
    write(name, fs, to_int16(data))


def rtf(start, end, w_len):
    return (end - start) / (w_len / fs)


def log_time(f, p1, p2):
    start_t = time.time()
    res = f(p1, p2)
    end_t = time.time()
    print(f"elapsed = {(end_t - start_t):5f}s")
    return res


def loadAM(amFile, dev):
    am_dir = os.path.dirname(amFile)
    mf = {'train_config': os.path.join(am_dir, 'config-run.yaml'),
          'model_file': amFile}
    text2speech = Text2Speech(
        **mf,
        device=dev,
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2
        speed_control_alpha=1.0,
    )
    text2speech.spc2wav = None  # Disable griffin-lim
    return text2speech


def loadVocoder(vocFile, dev):
    vocoder = load_model(vocFile).to(dev).eval()
    vocoder.remove_weight_norm()
    return vocoder


def synthesize(phones, am, voc):
    with torch.no_grad():
        start_am = time.time()
        am_res = am(phones)
        end_am = time.time()
        wav = voc.inference(am_res["feat_gen"])
        end_voc = time.time()
    print(f"RTF all = {rtf(start_am, end_voc, len(wav)):5f}, time = {(end_voc - start_am):5f}s")
    print(f"RTF am  = {rtf(start_am, end_am, len(wav)):5f}, time = {(end_am - start_am):5f}s")
    print(f"RTF voc = {rtf(end_am, end_voc, len(wav)):5f}, time = {(end_voc - end_am):5f}s")
    return wav.view(-1).cpu().numpy()


def main(argv):
    parser = argparse.ArgumentParser(description="Synthesizes wav file from phones",
                                     epilog="E.g. cat input.txt | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default='', type=str, help="Output File", required=True)
    parser.add_argument("--am", default='', type=str, help="AM File", required=True)
    parser.add_argument("--voc", default='', type=str, help="Vocoder File", required=True)
    parser.add_argument("--dev", default='cpu', type=str, help="Device: cpu | cuda | cuda:1", required=False)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    lines = []
    for line in sys.stdin:
        s_line = line.strip()
        lines.append(s_line)
    phones = " ".join(lines).strip()

    print("Phones: == %s ==" % phones, file=sys.stderr)
    print("Loading AM from : %s" % args.am, file=sys.stderr)
    am = log_time(loadAM, args.am, args.dev)
    print("Loading Vocoder from : %s" % args.voc, file=sys.stderr)
    voc = log_time(loadVocoder, args.voc, args.dev)
    print("Synthesizing...", file=sys.stderr)
    data = synthesize(phones, am, voc)
    print("Saving audio", file=sys.stderr)
    write_wav(args.out, data)

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

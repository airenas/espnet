import argparse
import logging
import os
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Take audio from dir and prepare for MOS evaluation. Split audio into segments based on duration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, help="Audio_input dir")
    parser.add_argument("--output", type=Path, help="Audio_output dir")
    parser.add_argument("--segment-duration", type=float, default=20.0,
                        help="Duration of each audio segment in seconds")
    parser.add_argument("--total_audio_seconds", type=float, default=600.0,
                        help="Total duration of audio to process in seconds")                        
    args = parser.parse_args()

    logging.info(f"input {args.input}")
    logging.info(f"output {args.output}")
    logging.info(f"segment duration {args.segment_duration}")
    logging.info(f"total audio seconds {args.total_audio_seconds}")

    if not args.output.exists():
        args.output.mkdir(parents=True)
        logging.info(f"Created output directory {args.output}")

    if not args.input.exists():
        logging.error(f"Input directory {args.input} does not exist")
        return

    audio_files = list(args.input.glob("*.wav"))
    if not audio_files:
        logging.error(f"No audio files found in {args.input}")
        return

    total_audio_ms = int(args.total_audio_seconds * 1000)
    processed_audio_ms = 0
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        if processed_audio_ms >= total_audio_ms:
            break
        output_file = args.output / audio_file.name
        if not output_file.exists():
            # Split audio into segments based on duration
            audio = AudioSegment.from_wav(audio_file)
            segment_duration_ms = int(args.segment_duration * 1000)
            for i, start in enumerate(range(0, len(audio), segment_duration_ms)):
                if processed_audio_ms >= total_audio_ms:
                    break
                segment = audio[start:start + segment_duration_ms]
                if len(segment) < 2000:
                    break
                segment.export(args.output / f"{audio_file.stem}_segment{i}.wav", format="wav")
                logging.info(f"Exported segment {i} of {audio_file.name}")
                processed_audio_ms += len(segment)

    logging.info(f"Taken total of {processed_audio_ms / 1000:.2f} seconds of audio")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")

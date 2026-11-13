import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pyworld
from tqdm import tqdm
import soundfile as sf


def to_s(min_d, freq, shift):
    t_shift = shift / freq
    return min_d * t_shift


def main(argv):
    parser = argparse.ArgumentParser(description="Detect long durations",
                                     epilog="E.g. cat durations | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--duration", default=150, type=int, help="Duration len in points to indicate a problem",
                        required=False)
    parser.add_argument("--freq", default=22050, type=int, help="Point in one sec. Used to calculate shift duration",
                        required=False)
    parser.add_argument("--shift", default=256, type=int, help="Shift in points. Used to calculate shift duration",
                        required=False)
    parser.add_argument("--input-durations", type=Path, help="Durations file",
                        required=True)
    parser.add_argument("--input-csv", type=Path, help="CSV file to contain the transcripts",
                        required=True)
    parser.add_argument("--wav-dir", type=Path, help="Directory containing WAV files",
                        required=True)
    parser.add_argument("--calc-dio", type=int, help="Whether to calculate DIO",
                        required=False, default=1)

    args = parser.parse_args(args=argv)

    logging.info(f"Inspecting durations with duration={args.duration}, freq={args.freq}, shift={args.shift}")
    logging.info(f"Input durations file: {args.input_durations}")
    logging.info(f"Input CSV file: {args.input_csv}")
    logging.info(f"WAV directory: {args.wav_dir}")
    logging.info(f"Calc dio: {args.calc_dio}")

    names, durations, transcripts = [], [], []

    with open(args.input_durations, "r", encoding="utf-8") as f_dur:
        with open(args.input_csv, "r", encoding="utf-8") as f_csv:
            for l_dur, l_csv in tqdm(zip(f_dur, f_csv), desc="Checking durations"):
                if l_dur.strip() == "" and l_csv.strip() == "":
                    continue
                durs = l_dur.split()
                d_name = durs[0]
                d_values = [int(x) for x in durs[1:]]
                trans = l_csv.split("|")
                t_name = trans[0].strip()
                phonemes = trans[3].strip().split()
                if d_name != t_name:
                    logging.error(f"Name mismatch: {d_name} != {t_name}")
                if len(d_values) != (len(phonemes) + 1):
                    logging.error(
                        f"Length mismatch for {d_name}: {len(d_values)} durations vs {len(phonemes)} phonemes")
                names.append(d_name)
                durations.append(d_values)
                transcripts.append(phonemes)

                # wav = args.wav_dir / f"{d_name}.wav"
                # if not wav.exists():
                #     logging.warning(f"WAV file not found: {wav}")
    for name, dur_values, phonemes in tqdm(zip(names, durations, transcripts), desc="Phoneme duration check", total=len(names)):
        for i, (dur, phoneme) in enumerate(zip(dur_values[:-1], phonemes)):
            if dur > args.duration:
                logging.error(
                    f"Long duration detected: {name}, phoneme {phoneme} at index {i} has duration {dur} points")

    for name, dur_values in tqdm(zip(names, durations), desc="Test wav len", total=len(names)):
        d = sum(dur_values)
        d_frames = int(d * args.shift)

        wav = args.wav_dir / f"{name}.wav"
        if not wav.exists():
            logging.error(f"WAV file not found: {wav}")
            continue

        frames = sf.info(wav).frames
        if (d_frames - frames) > args.shift:
            logging.error(
                f"WAV file {wav} is wrong durations: calculated({d_frames}) frames vs wav({frames}) frames. Difference: {d_frames - frames} frames")
        if (frames - d_frames) >= 0:
            logging.error(
                f"WAV file {wav} is wrong durations: calculated({d_frames}) frames vs wav({frames}) frames. Want positive difference")


    if args.calc_dio != 1:
        logging.info("Skipping DIO calculation as per --calc-dio=False")
    else:
        logging.info("Starting DIO calculation for duration verification")
        for name, dur_values in tqdm(zip(names, durations), desc="Test wav len (Dio exact)", total=len(names)):
            wav = args.wav_dir / f"{name}.wav"
            if not wav.exists():
                logging.error(f"WAV file not found: {wav}")
                continue

            try:
                # Read audio and extract Dio frames exactly like ESPnet does
                x, fs = sf.read(wav)
                if fs != args.freq:
                    logging.warning(f"Sample rate mismatch for {name}: wav fs={fs} vs args.freq={args.freq}")

                # Run pyworld.dio to get exact expected frame count
                f0, _ = pyworld.dio(
                    x.astype(np.float64),
                    fs,
                    f0_floor=40,
                    f0_ceil=240,
                    frame_period=(args.shift / fs) * 1000
                )
                target_frames = len(f0)
                duration_sum = sum(dur_values)

                # ESPnet assertion: 0 <= target_frames - duration_sum < reduction_factor (1)
                # Which means duration_sum must equal target_frames exactly
                if duration_sum != target_frames:
                    diff = target_frames - duration_sum
                    logging.error(
                        f"Duration mismatch for {name}: sum({duration_sum}) != Dio frames({target_frames}). Difference: {diff} frames"
                    )
            except Exception as e:
                logging.error(f"Failed processing {wav} with pyworld.dio: {e}")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))
    logging.info(f"Starting")

    main(sys.argv[1:])

    logging.info(f"Done")

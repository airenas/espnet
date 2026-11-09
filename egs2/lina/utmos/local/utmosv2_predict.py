import argparse
import csv
import logging
import os
import sys
from contextlib import nullcontext
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import utmosv2
from tqdm import tqdm


TARGET_SR = 16000


def load_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    return x, TARGET_SR


def device():
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda:
        return f"cuda:{cuda}"
    return "cpu"
     

def main():
    parser = argparse.ArgumentParser(description="Calculates utmos v2 for files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", nargs="*", help="Audio files to process")
    parser.add_argument("--out", help="CSV output file (default: stdout)")
    parser.add_argument("--eval-sentences", type=int, default=0, help="Number of sentences to evaluate (0 for all)")
    args = parser.parse_args()

    files = args.files
    if not files:
        parser.error("provide at least one audio file")

    model = utmosv2.create_model(pretrained=True, fold=2)

    output = open(args.out, "w", newline="", encoding="utf-8") if args.out else nullcontext(sys.stdout)
    with output as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", "sampling_rate", "mos"])
        for audio_file in tqdm(files[:args.eval_sentences] if args.eval_sentences > 0 else files, desc="Calculate UTMOS v2"):
            data, sampling_rate = load_16k(audio_file)
            mos = model.predict(data=data, sr=sampling_rate, device=device(), num_repetitions=5)
            mos = float(np.asarray(mos).reshape(-1)[0])
            writer.writerow([audio_file, sampling_rate, f"{mos:.4f}"])
            logging.info(f"{audio_file} {mos}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")

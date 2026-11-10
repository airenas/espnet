import argparse
import csv
import logging
import os
import sys
from contextlib import nullcontext

import librosa
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Calculates utmos for files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", nargs="*", help="Audio files to process")
    parser.add_argument("--out", help="CSV output file (default: stdout)")
    parser.add_argument("--eval-sentences", type=int, default=0, help="Number of sentences to evaluate (0 for all)")
    args = parser.parse_args()

    files = args.files
    if not files:
        parser.error("provide at least one audio file")

    logging.info(f"Calculating UTMOS for {len(files)} files")
    model_repo = "tarepan/SpeechMOS:v1.2.0"
    logging.info(f"Loading UTMOS model {model_repo}")
    predictor = torch.hub.load(model_repo, "utmos22_strong", trust_repo=True)

    output = open(args.out, "w", newline="", encoding="utf-8") if args.out else nullcontext(sys.stdout)
    with output as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", "sampling_rate", "mos"])
        for audio_file in tqdm(files[:args.eval_sentences] if args.eval_sentences > 0 else files,
                               desc="Calculate UTMOS"):
            wave, sr = librosa.load(audio_file, sr=None, mono=True)
            with torch.no_grad():
                mos = predictor(torch.from_numpy(wave).unsqueeze(0), sr)
            mos = float(np.asarray(mos.detach().cpu()).reshape(-1)[0])
            writer.writerow([audio_file, sr, f"{mos:.4f}"])
            logging.info(f"{audio_file} {mos}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")

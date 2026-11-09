import argparse
import csv
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Synthesize audio",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, help="CSV file with MOS")
    args = parser.parse_args()

    logging.info(f"input {args.input}")

    durations = []

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if row:
                durations.append(float(row[2]))

    min_val = min(durations)
    max_val = max(durations)
    avg_val = sum(durations) / len(durations)

    logging.info(f"Test: {len(durations)}")
    logging.info(f"Min: {min_val:.4f}")
    logging.info(f"Max: {max_val:.4f}")
    logging.info(f"Avg: {avg_val:.4f}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")

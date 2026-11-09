import argparse
import csv
import logging
import os
from pathlib import Path
import statistics  # Built-in module for median and statistics


def main():
    parser = argparse.ArgumentParser(description="Calculate MOS statistics",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, help="CSV file with MOS")
    args = parser.parse_args()

    logging.info(f"input {args.input}")

    scores = []

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if row:
                scores.append(float(row[2]))

    if not scores:
        logging.warning("No scores found in input file.")
        return

    # Sort scores for percentile and trimmed mean calculations
    scores_sorted = sorted(scores)
    n = len(scores_sorted)

    min_val = scores_sorted[0]
    max_val = scores_sorted[-1]
    avg_val = sum(scores_sorted) / n
    median_val = statistics.median(scores_sorted)

    trim_ratio = 0.10
    k = int(n * trim_ratio)
    
    # Ensure we don't trim everything if the list is very short
    if n - 2 * k > 0:
        trimmed_scores = scores_sorted[k : n - k]
        trimmed_mean_val = sum(trimmed_scores) / len(trimmed_scores)
    else:
        trimmed_mean_val = avg_val  # Fallback if too few items

    logging.info(f"Total files: {n}")
    logging.info(f"Min: {min_val:.4f}")
    logging.info(f"Max: {max_val:.4f}")
    logging.info(f"Avg: {avg_val:.4f}")
    logging.info(f"Trimmed Mean (10%): {trimmed_mean_val:.4f}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")
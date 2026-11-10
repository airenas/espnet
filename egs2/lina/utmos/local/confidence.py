import argparse
import csv
import logging
import os
from pathlib import Path
import statistics
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description="Calculate confidence interval and statistics of MOS differences between two systems",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input1", type=Path, required=True, help="Baseline CSV file with MOS")
    parser.add_argument("--input2", type=Path, required=True, help="Comparison CSV file with MOS")
    args = parser.parse_args()

    logging.info(f"Input 1 (Baseline): {args.input1}")
    logging.info(f"Input 2 (Comparison): {args.input2}")

    # Load baseline scores mapped by filename
    scores1 = {}
    with open(args.input1, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if row:
                filename = Path(row[0]).name
                scores1[filename] = float(row[2])

    # Load comparison scores mapped by filename
    scores2 = {}
    with open(args.input2, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if row:
                filename = Path(row[0]).name
                scores2[filename] = float(row[2])

    # Check that both inputs contain the exact same set of files
    keys1 = set(scores1.keys())
    keys2 = set(scores2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            logging.error(f"Files present in Input 1 but missing in Input 2: {sorted(missing_in_2)}")
        if missing_in_1:
            logging.error(f"Files present in Input 2 but missing in Input 1: {sorted(missing_in_1)}")
        raise ValueError("Input CSV files do not contain the exact same set of audio files. Comparison failed.")

    # Sort filenames to guarantee correct paired alignment regardless of row order in CSVs
    sorted_files = sorted(keys1)

    vals1 = [scores1[fname] for fname in sorted_files]
    vals2 = [scores2[fname] for fname in sorted_files]
    diffs = [v2 - v1 for v1, v2 in zip(vals1, vals2)]
    
    n = len(diffs)
    mean1 = statistics.mean(vals1)
    mean2 = statistics.mean(vals2)
    mean_diff = statistics.mean(diffs)
    stdev_diff = statistics.stdev(diffs) if n > 1 else 0.0
    se_diff = stdev_diff / (n ** 0.5) if n > 0 else 0.0

    # Compute exact t-test and confidence intervals via scipy
    t_stat, p_val = stats.ttest_rel(vals2, vals1)
    ci = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se_diff)

    logging.info(f"Matched samples: {n}")
    logging.info(f"Mean Difference (Input 2 - Input 1): {mean_diff:.4f}")

    print("\n=== MOS Difference Evaluation ===")
    print(f"Total Paired Samples : {n}")
    print(f"Baseline Mean (1)    : {mean1:.4f}")
    print(f"Comparison Mean (2)  : {mean2:.4f}")
    print(f"Mean Difference (2-1): {mean_diff:.4f}")
    print(f"Standard Deviation   : {stdev_diff:.4f}")
    print(f"Standard Error (SE)  : {se_diff:.4f}")
    print(f"95% Confidence Int.  : [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Paired t-statistic   : {t_stat:.4f}")
    print(f"Paired p-value       : {p_val:.4e}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info("Starting difference analysis")

    main()

    logging.info("Done")
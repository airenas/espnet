import argparse
import csv
import logging
import os
import re


def main(args):
    file_path = args.input
    if not os.path.isfile(file_path):
        logging.error(f"File {file_path} does not exist.")
        return

    total_records = 0
    total_words = 0
    total_chars = 0
    unique_words = set()

    word_pattern = re.compile(r"\w+", re.UNICODE)

    with open(file_path, newline="", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            row =  line.split("|")
            if len(row) < 3:
                logging.warning(f"skip {row}")
                continue

            text = row[2].strip()

            total_records += 1
            total_chars += len(text)

            words = word_pattern.findall(text.lower())
            total_words += len(words)

            unique_words.update(words)

    avg_words = total_words / total_records if total_records else 0

    print()
    print(f"Įrašai           {total_records}")
    print(f"Žodžiai          {total_words}")
    print(f"Simboliai        {total_chars}")
    print(f"Kalbėtojai              1")
    print(f"Vidut. žodžių skaičius  {avg_words:.2f}")
    print(f"Skirtingų žodžių        {len(unique_words)}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input csv file. id | words | words", )
    args = parser.parse_args()

    logging.info(f"Starting")

    main(args)

    logging.info("Done")

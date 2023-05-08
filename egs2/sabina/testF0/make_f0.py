#!/usr/bin/env python3
import argparse
import sys


def get(line, first, t):
    if t == "first":
        return first
    if t == "inc":
        return float(line) + 0.1
    if t == "low":
        return float(line) - 0.1


def main(argv):
    print("Starting", file=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='', required=True)
    args = parser.parse_args(args=argv)

    first = ""
    for line in sys.stdin:
        if not first:
            first = line.strip()
        print(get(line, first, args.type))
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

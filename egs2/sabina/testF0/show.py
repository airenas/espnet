#!/usr/bin/env python3
import argparse
import sys

import numpy
from matplotlib import pyplot


def main(argv):
    print("Starting", file=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='file', required=True)
    args = parser.parse_args(args=argv)

    print("Read data from: %s" % args.data, file=sys.stderr)
    with open(args.data) as f:
        data = numpy.loadtxt(args.data, dtype=numpy.float32)
    print("Data: %s" % data, file=sys.stderr)

    pyplot.figure(1)
    pyplot.plot(data)

    pyplot.show()


if __name__ == "__main__":
    main(sys.argv[1:])

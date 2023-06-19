import argparse
import sys

import numpy as np


def main(argv):
    parser = argparse.ArgumentParser(description="Restores F0 for analysis",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default='', type=str, help="Input f0 file", required=True)
    parser.add_argument("--txts", default='', type=str, help="Input texts file", required=True)
    parser.add_argument("--durs", default='', type=str, help="Input durations file", required=True)
    parser.add_argument("--name", default='', type=str, help="Audio name in txt file", required=True)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    with open(args.txts) as f:
        lines = [line.rstrip('\n').split(' ') for line in f]
    ld = {li[0]: li[1:] for li in lines}
    txts = ld[args.name]

    with open(args.durs) as f:
        lines = [line.rstrip('\n').split(' ') for line in f]
    ld = {li[0]: li[1:] for li in lines}
    durs = ld[args.name]

    d = np.load(args.input)
    f0 = np.exp(d)
    print("Mean: {}, min: {}, max: {}".format(f0.mean(), f0.min(), f0.max()), file=sys.stderr)

    for i in range(len(f0) - 1):
        print("%.2f\t%d\t%s" % (f0[i], int(durs[i]), txts[i]))

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

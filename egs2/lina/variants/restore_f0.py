import argparse
import sys

import numpy as np

from chainerx import math


def main(argv):
    parser = argparse.ArgumentParser(description="Restores F0 for analysis",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default='', type=str, help="Output File", required=True)
    parser.add_argument("--in-f0", default='', type=str, help="Input f0 File", required=True)
    parser.add_argument("--norm", default='', type=str, help="Normalization file", required=True)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    npz = np.load(args.norm)
    mean = npz['sum'] / npz['count']
    var = npz['sum_square'] / npz['count'] - mean * mean
    std = math.sqrt(var)
    d = np.loadtxt(args.in_f0)
    f0 = np.exp(d * std + mean)
    print("Mean: {}, min: {}, max{}".format(f0.mean(), f0.min(), f0.max()), file=sys.stderr)

    np.savetxt(args.out, f0, delimiter=',', fmt="%.5f")
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

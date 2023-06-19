import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np


def main(argv):
    parser = argparse.ArgumentParser(description="Restores F0 for analysis",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default='', type=str, help="Input f0 file", required=True)
    parser.add_argument("--output", default='', type=str, help="Output image", required=True)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    with open(args.input) as f:
        lines = [line.rstrip('\n').split('\t') for line in f]

    res, x, lx = [], [], 0
    for line in lines:
        f0 = float(line[0])
        d = int(line[1])
        for i in range(d):
            res.append(f0)
            x.append(lx)
            lx += 256

    plt.plot(np.array(x)/22050, np.array(res))
    plt.savefig(args.output)

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

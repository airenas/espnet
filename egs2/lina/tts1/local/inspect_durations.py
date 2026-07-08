import argparse
import sys


def to_s(min_d, freq, shift):
    t_shift = shift / freq
    return min_d * t_shift


def main(argv):
    parser = argparse.ArgumentParser(description="Detect long durations",
                                     epilog="E.g. cat durations | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--duration", default=150, type=int, help="Duration len in points to indicate a problem",
                        required=False)
    parser.add_argument("--freq", default=22050, type=int, help="Point in one sec. Used to calculate shift duration",
                        required=True)
    parser.add_argument("--shift", default=256, type=int, help="Shift in points. Used to calculate shift duration",
                        required=True)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    i, p = 0, 0
    min_d, max_d, sum_d, n_d = 0, 0, 0, 0
    for d in sys.stdin.readlines():
        i += 1
        words = d.split()
        l = [k for k in words[1:]]
        il = [int(k) for k in l]
        gt = [k for k in il if k > args.duration]
        if len(gt) > 0:
            p += 1
            print(words[0], gt)

        duration = sum(il)
        sum_d += duration
        n_d += 1
        if min_d == 0 or duration < min_d:
            min_d = duration
        if duration > max_d:
            max_d = duration

    print("Read %d lines. Possible problems %d" % (i, p), file=sys.stderr)
    print(f"Min {to_s(min_d, args.freq, args.shift):.3}s, max: {to_s(max_d, args.freq, args.shift):.3}s, "
          f"avg = {to_s(sum_d / n_d, args.freq, args.shift):.3}s", file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import sys


def is_name(line):
    return line.startswith("\"")


def update(line, prefix, strip_prefix):
    line = line.lstrip("\"")
    if strip_prefix:
        line = line.lstrip(strip_prefix)

    return "\"" + prefix + line


def main(argv):
    parser = argparse.ArgumentParser(description="Change segment name - add speaker",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + " > result.mlf",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prefix", default='SAB_', type=str, help="Add prefix", required=False)
    parser.add_argument("--strip-prefix", default='', type=str, help="Strip lab prefix", required=False)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    print(f"Add prefix: {args.prefix}, strip prefix: {args.strip_prefix}", file=sys.stderr)

    lc = 0
    for line in sys.stdin:
        lc += 1
        line = line.rstrip()
        if is_name(line) and args.prefix:
            line = update(line, args.prefix, args.strip_prefix)
        print(line)
    print("Read %d lines" % lc, file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

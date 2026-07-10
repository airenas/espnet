import argparse
import fileinput
import sys


def is_name(line):
    return line.startswith("\"")


def update(line, prefix, strip_prefix):
    line = line.removeprefix("\"")
    if strip_prefix:
        line = line.removeprefix(strip_prefix)

    return "\"" + prefix + line


def main(argv):
    parser = argparse.ArgumentParser(description="Change segment name - add speaker",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + " > result.mlf",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prefix", default='SAB_', type=str, help="Add prefix", required=False)
    parser.add_argument("--strip-prefix", default='', type=str, help="Strip lab prefix", required=False)
    parser.add_argument("files", nargs="*", help="Input MLF file(s), reads stdin when omitted")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    print(f"Add prefix: {args.prefix}, strip prefix: {args.strip_prefix}", file=sys.stderr)

    lc = 0
    files = args.files if args.files else ("-",)
    for line in fileinput.input(files=files):
        lc += 1
        line = line.rstrip()
        if is_name(line) and args.prefix:
            line = update(line, args.prefix, args.strip_prefix)
        print(line)
    print("Read %d lines" % lc, file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import re
import sys


def ignore(line):
    return line == "#!MLF!#"


def get_fn(line):
    return line.lstrip("\"").rstrip("\"").replace(".lab", "")


def main(argv):
    parser = argparse.ArgumentParser(description="Excludes provided files",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + " > result.mlf",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exclude", default='', type=str, help="Exclude regexp", required=False)
    parser.add_argument("--excludeFromFile", default='', type=str, help="Exclude file list", required=False)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    es = set()
    if args.exclude:
        print("Will exclude %s " % args.exclude, file=sys.stderr)
        regexp = re.compile("\\s+")
        strs = regexp.split(args.exclude)
        es = set(strs)
    elif args.excludeFromFile:    
        with open(args.excludeFromFile) as file:
            for line in file:
                es.add(line.rstrip())
        print("Will exclude from %s %d items" % (args.excludeFromFile, len(es)), file=sys.stderr)        

    fc = 0
    ec = 0
    exclude = False
    for line in sys.stdin:
        line = line.rstrip()
        if ignore(line):
            print(line)
        elif line.startswith("\""):
            fc += 1
            name = get_fn(line)
            # print("Name '%s' " % name, file=sys.stderr)
            if name in es:
                ec += 1
                print("Excluding %s " % line, file=sys.stderr)
                exclude = True
            if not exclude:
                print(line)
        elif line == ".":
            if not exclude:
                print(line)
            exclude = False
        else:
            if not exclude:
                print(line)
    print("Read %d files, excluded %d" % (fc, ec), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

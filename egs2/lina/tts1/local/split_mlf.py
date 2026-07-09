import argparse
import sys
import os
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description="Splits mlf to many files",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outDir", default='', type=str, help="Output directory", required=True)                                     
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    file = None
    lc = 0
    fc = 0
    for line in sys.stdin:
        lc += 1
        sLine = line.strip()
        if (sLine == "#!MLF!#"):
            continue
        elif sLine.startswith("\""):
            fc += 1
            p = Path(sLine.strip('""')).name
            fn = os.path.join(args.outDir, p)
            print("Writing: %s" % fn , file=sys.stderr)
            file = open(fn ,'w', encoding='utf-8')
        elif sLine == ".":
            file.close()
            file = None
        else:
            print(line, end="", file=file)

    if not (file is None):
        file.close()

    print("Read %d lines, %d files" % (lc, fc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
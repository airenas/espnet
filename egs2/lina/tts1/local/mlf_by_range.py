import argparse
import os
import sys

import mlf
import range


def write_mlf(m, r, file):
    mc = mlf.MlfLine(m.from_, m.to, m.ph, m.syll, m.word, m.phr, m.sent, m.pos, m.punct)
    from_ = int(m.from_)
    to = int(m.to)
    from_ = from_ - r.from_
    to = to - r.from_
    if to > (r.to - r.from_):
        to = (r.to - r.from_)
    if from_ < 0:
        from_ = 0
    mc.from_ = str(from_).rjust(10)
    mc.to = str(to).rjust(10)
    print(mc.str(), file=file)
    pass


def mlf_in(m, r):
    from_ = int(m.from_)
    to = int(m.to)
    return (r.from_ <= from_ < r.to) or (r.from_ < to <= r.to)


def write_range(name, r, mlfs, file):
    # print("Write %s" % name, file=sys.stderr)
    print("\"%s\"" % name, file=file)
    for m in mlfs:
        if mlf_in(m, r):
            write_mlf(m, r, file)
    print(".", file=file)


def write_ranges(name, ranges, mlfs, file):
    sc = 0
    for r in ranges:
        write_range("%s_%03d.lab" % (name, sc), r, mlfs, file)
        sc += 1
    return sc


def main(argv):
    parser = argparse.ArgumentParser(description="Splits lab files into smaller by range",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rangeDir", default='', type=str, help="Directory for ranges", required=True)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    ranges = []
    mlfs = []
    name = ""
    sc = 0
    lc = 0
    for line in sys.stdin:
        lc += 1
        s_line = line.strip()
        if s_line == "#!MLF!#":
            continue
        elif s_line.startswith("\""):
            sc += write_ranges(name, ranges, mlfs, sys.stdout)
            name = s_line.strip('""').replace(".lab", "")
            ranges = range.load_ranges_file(os.path.join(args.rangeDir, name + ".split.range"))
            mlfs = []
        elif s_line == ".":
            continue
        else:
            m_line = mlf.from_str(line.rstrip())
            mlfs.append(m_line)

    sc += write_ranges(name, ranges, mlfs, sys.stdout)

    print("Read %d lines, %d splits" % (lc, sc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

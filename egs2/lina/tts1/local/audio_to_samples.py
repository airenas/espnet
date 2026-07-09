import argparse
import os
import sys

import sox


def main(argv):
    parser = argparse.ArgumentParser(description="Extract audio samples from file list",
                                     epilog="E.g. ls -1 *.wav | " + sys.argv[0] + " > audio.samples",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    lc = 0
    for line in sys.stdin:
        s_line = line.strip()
        if s_line:
            lc += 1
            n_samples = sox.file_info.num_samples(s_line)
            f_name, _ = os.path.splitext(os.path.basename(s_line))
            print("%s %d" % (f_name, n_samples), file=sys.stdout)

    print("Read %d lines" % (lc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

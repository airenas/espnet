import argparse
import sys

import mlf


class LastData:
    def __init__(self):
        self.punct = ""
        self.phones = []

    def add_to(self, phones, skip_sp):

        if len(self.phones) > 0:
            for ph in self.phones[:-1]:
                phones.append(ph)

            if self.punct != "":
                if mlf.is_sil(self.phones[-1]):
                    phones.append(self.punct)
                    if not (self.phones[-1] == "sp" and skip_sp):
                        phones.append(self.phones[-1])
                else:
                    phones.append(self.phones[-1])
                    phones.append(self.punct)
            else:
                phones.append(self.phones[-1])

        self.punct = ""
        self.phones = []


def get_punct(s):
    if s == "-":
        return " -"
    return s


def write_line(name, words, phones, file):
    if name != "":
        s = " ".join(words)
        ph = ""
        if len(phones) > 0:
            ph = "|" + " ".join(phones)
        print("%s|%s|%s%s" % (name, s, s.lower(), ph), file=file)


def main(argv):
    parser = argparse.ArgumentParser(description="Convert mlf to specific csv",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + " > result.mlf",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outputPhones", default=False, action='store_true', help="Do output phones")
    parser.add_argument("--skipSP", default=False, action='store_true', help="Skip 'sp' after punctuation")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    lc = 0
    wc = 0
    name = ""
    ld = LastData()
    words, phones = [], []
    for line in sys.stdin:
        lc += 1
        s_line = line.strip()
        if s_line == "#!MLF!#":
            continue
        elif s_line.startswith("\""):
            ld.add_to(phones, args.skipSP)
            write_line(name, words, phones, sys.stdout)
            name = s_line.strip('""').replace(".lab", "")
            words, phones, lp = [], [], ""
        elif s_line == ".":
            continue
        else:
            m_line = mlf.from_str(line.rstrip())
            if m_line.is_word():
                wc += 1
                words.append(m_line.word + get_punct(m_line.punct))
                if args.outputPhones:
                    ld.add_to(phones, args.skipSP)
                    ld.punct = m_line.punct
            if args.outputPhones:
                ld.phones.append(m_line.ph)
                # if m_line.is_sil():
                #     if not (m_line.ph == "sp" and args.skipSP and ld.punct != ""):
                #         ld.sil = m_line.ph
                # else:
                #     phones.append(m_line.ph)
    ld.add_to(phones, args.skipSP)
    write_line(name, words, phones, sys.stdout)

    print("Read %d lines, %d words" % (lc, wc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

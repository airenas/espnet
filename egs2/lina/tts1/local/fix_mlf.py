import argparse
import fileinput
import re
import sys

import mlf


def ignore(line):
    return line == "#!MLF!#" or line.startswith("\"") or line == "."


def trim_accent(line):
    return line.replace("4", "").replace("9", "").replace("3", "")


def fix_phone(line):
    if line == "sill":
        return "sil"
    if line.startswith("'") and line.endswith("'"):
        return line.strip("''")
    return line


regexp = re.compile("\\s+")


def parse_line(line):
    strs = regexp.split(line)
    strs[0] = strs[0].rjust(10, ' ')
    strs[1] = strs[1].rjust(10, ' ')
    res = mlf.MlfLine(strs[0], strs[1], fix_phone(strs[3]))
    if len(strs) > 4:
        res.syll = strs[4]
    if len(strs) > 5:
        res.word = trim_accent(strs[5])
    if res.syll == "sp" or res.syll == "sil":
        if len(strs) > 6:
            res.phr = strs[6]
        if len(strs) > 7:
            res.sent = strs[7]
    for s in strs[6:]:
        if s == "/":
            res.phr = s
        elif s == "//":
            res.sent = s
        elif s.startswith("<"):
            res.punct = s[1]
            if s.startswith("<.><.><.>"):
                res.punct = "..."
    return res


def fix_symbols(line):
    return line.replace("�", "-").replace("–", "-").replace("—", "-").replace("<\">", "").replace("<„>", "").replace(
        "<“>", "").replace("<>", "")


def fix_sil(lines):
    la = len(lines)
    if la > 2 and lines[la - 2].is_sil():
        if lines[la - 1].sent == "":
            lines[la - 2].sent = ""


def fix_sils(mlfs):
    prev = None
    for ml in mlfs:
        if ml.ph in ["sil", "sp"] and prev is not None and prev.punct in [".", "!", "?"]:
            ml.ph = "sil"
            ml.syll = "sil"
            ml.word = "sil"
            ml.sent = "sil"
        if ml.word != "":
            prev =  ml

    for ml in mlfs:
        if ml.syll == "sp" and ml.ph == "sil":
            ml.ph = "sp"


def fix_words(mlfs):
    for i in range(1, len(mlfs) - 1):
        ml = mlfs[i]
        ml_next = mlfs[i + 1]
        if not ml.is_sil() and ml.syll == "" and mlfs[i - 1].is_sil():
            if not ml_next.is_sil() and ml_next.syll != "":
                print("Fix words for %s\n from %s" % (ml.str(), ml_next.str()), file=sys.stderr)
                ml.syll = ml_next.syll
                ml.word = ml_next.word
                ml_next.syll = ""
                ml_next.word = ""


def fix_sp(mlfs):
    res = []
    for ml in mlfs:
        if not (ml.is_sil() and ml.to == ml.from_):
            res.append(ml)
    return res


def fix_lines(lines):
    fix_words(lines)
    res = fix_sp(lines)
    fix_sils(res)
    return res


def fix_ne_excl(ml):
    if ml.word == "ne" and ml.punct == "!":
        ml.punct = "."
    if ml.word == "keiks" and ml.punct == "!":
        ml.punct = "."
    if ml.word == "gulėtumėt" and ml.punct == "!":
        ml.punct = "."


def fix_parantheses(ml):
    if ml.punct == ")" or ml.punct == "(":
        ml.punct = ","
    if ml.punct == "\\":
        ml.punct = "."
    if ml.punct == ";":
        ml.punct = ","


def fix_dots(ml):
    if ml.punct == "...":
        ml.punct = "."


def main(argv):
    parser = argparse.ArgumentParser(description="Fix mlf file for hts",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + " > result.mlf",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", nargs="*", help="Input MLF file(s), reads stdin when omitted")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    lc = 0
    lines = []
    mlw = None
    files = args.files if args.files else ("-",)
    for line in fileinput.input(files=files):
        lc += 1
        line = line.strip()
        line = fix_symbols(line)
        if not ignore(line):
            ml = parse_line(line)
            fix_parantheses(ml)
            fix_dots(ml)
            lines.append(ml)
            if ml.punct != "":
                if mlw is not None:
                    mlw.punct = ml.punct
                    fix_ne_excl(mlw)
                else:
                    print("Nowhere to add punct " + line, file=sys.stderr)
                mlw = None
                ml.punct = ""
            if ml.is_word():
                mlw = ml
            fix_sil(lines)
        else:
            for ml in fix_lines(lines):
                print(ml.str())
            lines = []
            mlw = None
            print(line)

    for ml in fix_lines(lines):
        print(ml.str())
    print("Read %d lines" % lc, file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import logging
import sys
from typing import List

import mlf


def change_phone(p) -> str:
    if p == ".":  ## do not drop dot
        return p
    return p.replace("'", "").replace(".", "")


class Word:
    def __init__(self, word: str, punctuation: str):
        self.word = word
        self.punctuation = punctuation
        self.phones = []
        self.is_sil = False

    def phones_str(self) -> str:
        if not self.phones:
            return ""

        phones = self.phones[:]
        if mlf.is_sil(phones[-1]) and self.punctuation != "":
            phones.insert(-1, get_punct(self.punctuation))
        elif self.punctuation != "":
            phones.append(get_punct(self.punctuation))
        return " ".join(phones)

    def phones_sp(self) -> str:
        """
        Adds sp to every word
        """
        if not self.phones:
            return ""

        phones = self.phones[:]
        if mlf.is_sil(phones[-1]) and self.punctuation != "":
            phones.insert(-1, get_punct(self.punctuation))
        elif self.punctuation != "":
            phones.append(get_punct(self.punctuation))
        if phones[-1] != "sp" and phones[-1] != "sil":
            phones.append("sp")
        return " ".join(phones)

    def word_phones_str(self) -> str:
        if not self.phones:
            return ""

        phones = self.phones[:]
        if mlf.is_sil(phones[-1]) and self.punctuation != "":
            phones.insert(-1, get_punct(self.punctuation))
        elif self.punctuation != "":
            phones.append(get_punct(self.punctuation))
        if mlf.is_sil(phones[-1]):
            phones[-1] = " " + phones[-1]

        res = []
        for p in phones:
            pn = change_phone(p)
            if pn:
                res.append(pn)

        return "".join(res).strip()


def get_words(words: List[Word]) -> str:
    res, prev = "", ""
    for w in words:
        if w.is_sil:
            continue
        # print(f"({res}), ({w.word}), ({prev})", file=sys.stderr)
        res += prev + w.word + get_punct(w.punctuation)
        prev = " "
    return res


def get_phones(words: List[Word]) -> str:
    res = []
    for w in words:
        phones_str = w.phones_str()
        res.append(phones_str)
    return " ".join(res)

def get_phones_sp(words: List[Word]) -> str:
    res = []
    for w in words:
        phones_str = w.phones_sp()
        res.append(phones_str)
    return " ".join(res)

def get_word_phones(words: List[Word]) -> str:
    res = []
    for w in words:
        phones_str = w.word_phones_str()
        if phones_str:
            res.append(phones_str)
    return " ".join(res)


class Line:
    def __init__(self, name: str):
        self.name = name
        self.words: List[Word] = []

    def last_word(self) -> Word:
        if len(self.words) == 0:
            raise ValueError("no words")
        return self.words[-1]

    def to_str(self, _type: str):
        word_str = get_words(self.words)
        phones = ""
        if _type == "phones":
            phones = get_phones(self.words)
        elif _type == "word_phones":
            phones = get_word_phones(self.words).strip()
        elif _type == "phones_sp":
            phones = get_phones_sp(self.words)
        else:
            raise RuntimeError(f"unknow output type: '{_type}'")
        return f"{self.name}|{word_str}|{word_str.lower()}|{phones}"


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
    parser.add_argument("--input", dest="input_file", type=argparse.FileType("r"), default=sys.stdin,
                        help="Input MLF file; read stdin when omitted")
    parser.add_argument("--type", default="ph", type=str, help="Phones type")
    parser.add_argument("--skipSP", default=False, action='store_true', help="Skip 'sp' after punctuation")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    lc = 0
    wc = 0
    ln: Line | None = None
    for line in args.input_file:
        lc += 1
        s_line = line.strip()
        try:
            if s_line == "#!MLF!#":
                continue
            if s_line.startswith("\""):
                if ln:
                    print(ln.to_str(_type=args.type), file=sys.stdout)
                ln = Line(name=s_line.strip('""').replace(".lab", ""))
            elif s_line == ".":
                continue
            else:
                if not ln:
                    raise ValueError("no item header")
                m_line = mlf.from_str(line.rstrip())
                if m_line.is_word():
                    wc += 1
                    ln.words.append(Word(word=m_line.word, punctuation=m_line.punct))
                if len(ln.words) == 0 and m_line.ph == "sil":
                    w = Word(word="", punctuation="")
                    w.is_sil = True
                    ln.words.append(w)

                ln.last_word().phones.append(m_line.ph)
        except BaseException as ex:
            raise ValueError(f"{ex}. Txt: {s_line}")
    if ln:
        print(ln.to_str(_type=args.type), file=sys.stdout)

    print("Read %d lines, %d words" % (lc, wc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

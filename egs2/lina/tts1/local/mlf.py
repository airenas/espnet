import re

regexp = re.compile("\\s")


def from_str(line):
    from_ = line[:10]
    to = line[11:21]
    strs = regexp.split(line[22:])
    return MlfLine(from_, to, strs[0], get(strs, 1), get(strs, 2), get(strs, 3), get(strs, 4), get(strs, 8),
                   get(strs, 9))


def get(arr, pos):
    if len(arr) > pos:
        return arr[pos]
    return ""


def is_sil(ph):
    return ph == "sp" or ph == "sil"


class MlfLine:
    def __init__(self, from_, to, ph, syll="", word="", phr="", sent="", pos="", punct=""):
        self.punct = punct
        self.from_ = from_
        self.to = to
        self.ph = ph
        self.syll = syll
        self.word = word
        self.phr = phr
        self.sent = sent
        self.pos = pos

    def str(self):
        return self.from_ + " " + self.to + " " \
               + (self.ph + " " + self.syll
                  + " " + self.word + " " + self.phr + " " + self.sent + " " + " " + " "
                  + " " + self.pos + " " + self.punct).strip()

    def is_word(self):
        return self.word != "" and self.word != "sp" and self.word != "sil" and self.word != "noise"

    def is_sil(self):
        return is_sil(self.ph)

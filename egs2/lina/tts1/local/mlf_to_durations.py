import argparse
import sys
from typing import List, Tuple

import mlf
from egs2.lina.tts1.local.mlf_to_csv import get_punct


def calc_duration(to, fr, shift, freq):
    to_ms = int(to) / 10000.0
    t_shift = 1000.0 * shift / freq
    return int(round((to_ms - fr * t_shift) / t_shift))


def fix_duration(v, v_max, shift):
    if v * shift <= v_max:
        return 1
    return 0


def get_dur_str(durations):
    res = ""
    p = ""
    for d in durations:
        res = res + p + str(d)
        p = " "
    return res


class Config:
    def __init__(self, shift, freq, punct_duration: int = 10):
        self.freq = freq
        self.shift = shift
        self.punct_duration = punct_duration


def calc_time(to, cfg: Config):
    to_ms = int(to) / 10000.0
    t_shift = 1000.0 * cfg.shift / cfg.freq
    return int(to_ms / t_shift)


class Phone:
    def __init__(self, phone: str, from_: int, to: int):
        self.phone = phone
        self.from_ = from_
        self.to = to


class Word:
    def __init__(self, word: str, punctuation: str):
        self.word = word
        self.punctuation = punctuation
        self.phones: List[Phone] = []
        self.is_sil = False

    def durations(self, from_: int, cfg: Config) -> Tuple[int, List[int]]:
        if not self.phones:
            return from_, []
        next_from, last_phone = from_, ""
        res = []
        for p in self.phones:
            phone_to = calc_time(p.to, cfg)
            dur = phone_to - next_from
            res.append(dur)
            next_from = phone_to
            last_phone = p.phone

        if self.punctuation != "":
            if mlf.is_sil(last_phone):
                d = cfg.punct_duration
                if res[-1] < d:
                    d = res[-1]
                res.insert(-1, d)
                res[-1] = res[-1] - d
            else:
                res.append(0)  # add zero for punctuation
        return next_from, res

    def durations_sp(self, from_: int, cfg: Config) -> Tuple[int, List[int]]:
        if not self.phones:
            return from_, []
        next_from, last_phone = from_, ""
        res = []
        for p in self.phones:
            phone_to = calc_time(p.to, cfg)
            dur = phone_to - next_from
            res.append(dur)
            next_from = phone_to
            last_phone = p.phone

        if self.punctuation != "":
            if mlf.is_sil(last_phone):
                d = cfg.punct_duration
                if res[-1] < d:
                    d = res[-1]
                res.insert(-1, d)
                res[-1] = res[-1] - d
            else:
                res.append(0)
        if not mlf.is_sil(last_phone):
            res.append(0)
        return next_from, res


def get_words(words: List[Word]) -> str:
    res, prev = "", ""
    for w in words:
        if w.is_sil:
            continue
        # print(f"({res}), ({w.word}), ({prev})", file=sys.stderr)
        res += prev + w.word + get_punct(w.punctuation)
        prev = " "
    return res


def get_simple_durations(words: List[Word], cfg: Config):
    from_, res = 0, []
    for w in words:
        from_, durations = w.durations(from_=from_, cfg=cfg)
        for d in durations:
            res.append(d)
    return res


def get_durations_sp(words: List[Word], cfg: Config):
    from_, res = 0, []
    for w in words:
        from_, durations = w.durations_sp(from_=from_, cfg=cfg)
        for d in durations:
            res.append(d)
    return res


class Line:
    def __init__(self, name: str, duration: int, cfg: Config):
        self.name = name
        self.words: List[Word] = []
        self.duration = duration
        self.cfg = cfg

    def last_word(self) -> Word:
        if len(self.words) == 0:
            raise ValueError("no words")
        return self.words[-1]

    def to_str(self, _type: str):
        if _type == "phones":
            durations = get_simple_durations(self.words, self.cfg)
        # elif _type == "word_phones":
        #     phones = get_word_phones(self.words).strip()
        elif _type == "phones_sp":
            durations = get_durations_sp(self.words, self.cfg)
        else:
            raise RuntimeError(f"unknow output type: '{_type}'")

        lf = sum(durations)
        durations[-1] += fix_duration(lf, self.duration, self.cfg.shift)
        dur_str = get_dur_str(durations)
        return f"{self.name} {dur_str} 0"


def main(argv):
    parser = argparse.ArgumentParser(description="Convert mlf to durations for fastspeech2 training",
                                     epilog="E.g. cat input.mlf | " + sys.argv[0] + " > result.mlf",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input_file", type=argparse.FileType("r"), default=sys.stdin,
                        help="Input MLF file; read stdin when omitted")
    parser.add_argument("--freq", default=22050, type=int, help="Point in one sec. Used to calculate shift duration",
                        required=True)
    parser.add_argument("--shift", default=256, type=int, help="Shift in points. Used to calculate shift duration",
                        required=True)
    parser.add_argument("--samplesFile", default='', type=str, help="File containing file samples", required=True)
    parser.add_argument("--type", default="ph", type=str, help="Phones type")
    parser.add_argument("--punct-duration", default=10, type=int, help="Punctuation duration")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    print(f"type {args.type}", file=sys.stderr)
    print(f"shift {args.shift}", file=sys.stderr)
    print(f"freq {args.freq}", file=sys.stderr)
    print(f"punct-duration {args.punct_duration}", file=sys.stderr)
    cfg = Config(shift=args.shift, freq=args.freq, punct_duration=args.punct_duration)
    print("Reading samples file : %s" % args.samplesFile, file=sys.stderr)
    samples = {}
    with open(args.samplesFile) as f:
        lines = f.readlines()
        for line in lines:
            s_line = line.strip()
            if s_line:
                ls = s_line.split(' ')
                samples[ls[0]] = int(ls[1])
    print("Read samples file with %d lines" % len(samples), file=sys.stderr)

    lc, wc = 0, 0
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
                name = s_line.strip('""').replace(".lab", "")
                ln = Line(name=name, duration=samples[name], cfg=cfg)
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
                ln.last_word().phones.append(Phone(m_line.ph, m_line.from_, m_line.to))

        except BaseException as ex:
            raise ValueError(f"{ex}. Txt: {s_line}")
    if ln:
        print(ln.to_str(_type=args.type), file=sys.stdout)

    print("Read %d lines, %d words" % (lc, wc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

class Range:
    def __init__(self, from_, to):
        self.from_ = from_
        self.to = to


def load_ranges_file(name):
    with open(name, "rt") as f:
        return load_ranges(f)


def load_ranges(file):
    res = []
    for line in file:
        line = line.strip()
        strs = line.split(" ")
        res.append(Range(int(strs[0]), int(strs[1])))
    return res

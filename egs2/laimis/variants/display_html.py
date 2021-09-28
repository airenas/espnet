import argparse
import os
import sys


def add_script():
    res = "<script>\n" \
          "function updateView(key, el, ons){" \
          "let s = el.innerHTML;" \
          "let root = document.documentElement;" \
          "if (s.startsWith(\"Hide\")) { " \
          "el.innerHTML = s.replace(\"Hide\", \"Show\"); " \
          "root.style.setProperty(\"--\" + key, \"none\"); " \
          "} else {" \
          "el.innerHTML = s.replace(\"Show\", \"Hide\"); " \
          "root.style.setProperty(\"--\" + key, ons);" \
          "} } </script>"
    return res


def exists_file(f):
    print("File %s - exists: %s" % (f, os.path.isfile(f)) , file=sys.stderr)
    return os.path.isfile(f)


def add_styles(vocs, ams):
    res = "<style>"
    res += ":root\n{\n"
    for k in vocs:
        res += "  --" + vocs.get(k) + ": none;\n"
    for k in ams:
        res += "  --" + ams.get(k) + ": none;\n"
    res += "}\n"
    for k in vocs:
        res += "  ." + vocs.get(k) + " { display: var(--" + vocs.get(k) + ");}\n"
    for k in ams:
        res += "  ." + ams.get(k) + " { display: var(--" + ams.get(k) + ");}\n"

    res += "</style>"
    return res


def add_table(name, d, ons):
    res = name
    res += "<br/><table><tbody>"
    keys = sorted(d.keys())
    for k in keys:
        res += "<tr><td> "
        res += "<button type=\"button\" onclick=\"updateView('" + d.get(
            k) + "', this, '" + ons + "')\">Show " + k + "</button>"
        res += "</td></tr>"
    res += "</tbody></table><br/>"
    return res


def get_sentence_html(s, vocabs, ams):
    gt_fn = os.path.join("mp3s", s.get("name") + "_gt.mp3")
    if exists_file(gt_fn):
        res = s.get("name") + " GT: <audio controls preload=\"none\"><source src=\"" + gt_fn + \
          "\" type=\"audio/mpeg\"></audio><br/>\n"
    else:
        res = s.get("name") + " GT: -<br/>\n"      
    res += "<table><thead><tr><th></th>"
    vocs = sorted(s.get("vocs"))
    for voc in vocs:
        res += "<th class=\"" + vocabs.get(voc) + "\"> " + voc + "</th>"
    res += "</tr></thead><tbody>"
    s_models = sorted(s.get("models"))
    models = s.get("modelD")
    for am in s_models:
        res += "<tr class=\"" + ams.get(am) + "\"><th>" + am + "</th>"
        for voc in vocs:
            res += "<td class=\"" + vocabs.get(voc) + "\"> "
            data = models.setdefault(am, {})
            v_data = data.setdefault(voc, {})
            file = v_data.get("file")
            if file is None:
                res += "-"
            else:
                res += "<audio controls preload=\"none\"><source src=\"" + file + "\" type=\"audio/mpeg\"></audio>"
            res += "</td>"
        res += "</tr>"

    res += "</tbody></table><br/>"
    return res


def main(argv):
    parser = argparse.ArgumentParser(description="Generates HTML file with sample audios",
                                     epilog="E.g. cat input.txt | " + sys.argv[0] + " > result.html",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    sentences = {}
    vocs = {}
    ams = {}
    for line in sys.stdin:
        s_line = line.strip()
        fa = os.path.basename(s_line)
        fn = os.path.splitext(fa)[0]
        spl = fn.split("_")

        sent = spl[0]
        am = spl[1]
        voc = spl[2]
        if voc == "5best":
            voc = spl[3]
        if vocs.get(voc) is None:
            vocs[voc] = "v-" + str(len(vocs))
        if ams.get(am) is None:
            ams[am] = "a-" + str(len(ams))
        sentence = sentences.setdefault(sent, {})
        sentence.setdefault("models", set()).add(am)
        sentence.setdefault("vocs", set()).add(voc)
        sentence.setdefault("name", sent)
        models = sentence.setdefault("modelD", {})
        data = models.setdefault(am, {})
        v_data = data.setdefault(voc, {})
        v_data["voc"] = voc
        v_data["file"] = s_line

    print("Preparing HTML...", file=sys.stderr)
    print("<html><head>")
    print(add_styles(vocs, ams))
    print(add_script())
    print("</head><body>")
    print(add_table("Vocoders", vocs, "table-cell"))
    print(add_table("Acoustic models", ams, "table-row"))
    for s in sentences:
        print(get_sentence_html(sentences[s], vocs, ams))
    print("</body></html>")
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
import argparse
import base64
import io
import sys
import time
from argparse import Namespace

import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import


def main(argv):
    print("Starting", file=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='text to be synthesized', required=True)
    parser.add_argument('--model_dir', type=str, help='Model dir', required=True)
    parser.add_argument('--out', type=str, help='Spectrogram output file', required=True)
    args = parser.parse_args(args=argv)

    print("Read data from: %s" % args.data, file=sys.stderr)
    with open(args.data) as f:
        data = f.read().rstrip("\n")
    print("Data: %s" % data, file=sys.stderr)

    dict_path = args.model_dir + "/vocab"
    model_path = args.model_dir + "/model.loss.best"

    print("Vocab: %s" % dict_path, file=sys.stderr)
    with open(dict_path) as f:
        lines = f.readlines()
    lines = [line.replace("\n", "").split(" ") for line in lines]
    char_to_id = {c: int(i) for c, i in lines}
    print("Vocab read: %d items" % len(char_to_id), file=sys.stderr)

    print("Model: %s" % model_path, file=sys.stderr)
    device = torch.device("cpu")
    idim, odim, train_args = get_model_conf(model_path)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    torch_load(model_path, model)
    model = model.eval().to(device)
    inference_args = Namespace(**{"threshold": 0.5, "minlenratio": 0.0, "maxlenratio": 10.0})
    print("Model loaded - now ready to synthesize!")

    def frontend(text):
        charseq = text.split(" ")
        idseq = []
        for c in charseq:
            if c.isspace():
                idseq += [char_to_id["<space>"]]
            elif c not in char_to_id.keys():
                idseq += [char_to_id["<unk>"]]
            else:
                idseq += [char_to_id[c]]
        idseq += [idim - 1]  # <eos>
        return torch.LongTensor(idseq).view(-1).to(device)

    with torch.no_grad():
        start = time.time()
        x = frontend(data)
        print(f"x = {x}")
        c, _, _ = model.inference(x, inference_args)
        am_start = time.time()
        elapsed = (am_start - start)
        print(f"acoustic model done: {elapsed:5f} s")
    buffer = io.BytesIO()
    torch.save(c, buffer)
    buffer.seek(0)
    encoded_data = base64.b64encode(buffer.read())
    base64_data = encoded_data.decode('ascii')
    print("Write to: %s" % args.out, file=sys.stderr)
    with open(args.out, "w") as text_file:
        text_file.write(base64_data)
    print(f"Len bytes: {len(base64_data):d}")
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

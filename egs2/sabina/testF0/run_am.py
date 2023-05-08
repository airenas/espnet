#!/usr/bin/env python3
import argparse
import base64
import io
import sys
import time

import torch
from espnet_model_zoo.downloader import ModelDownloader

from espnet2.bin.tts_inference import Text2Speech


def extract_model(model_zip_path):
    print("Model zip path: %s" % model_zip_path)
    d = ModelDownloader("~/.cache/espnet")
    m_extracted = d.unpack_local_file(model_zip_path)
    print("Model extraction info: %s" % m_extracted)
    return m_extracted


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

    model_dir = extract_model(args.model_dir)
    tts = Text2Speech(**model_dir,
                      device="cpu",
                      # Only for Tacotron 2\n",
                      threshold=0.5,
                      minlenratio=0.0,
                      maxlenratio=10.0,
                      use_att_constraint=True,
                      backward_window=1,
                      forward_window=3,
                      # Only for FastSpeech & FastSpeech2\n",
                      speed_control_alpha=1.0,
                      )
    tts.vocoder = None
    print("Model loaded - now ready to synthesize!")
    with torch.no_grad():
        start = time.time()
        res = tts(text=data, decode_conf={"alpha": 0.5})
        y = res["feat_gen"]
        end = time.time()
        elapsed = (end - start)
        print(f"acoustic model done: {elapsed:5f} s")
    buffer = io.BytesIO()
    torch.save(y, buffer)
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

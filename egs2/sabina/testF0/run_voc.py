#!/usr/bin/env python3
import argparse
import base64
import io
import sys
import time

import torch
import soundfile
from parallel_wavegan.utils import load_model


sampling_rate=22050

def main(argv):
    print("Starting", file=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Spectrogram input file', required=True)
    parser.add_argument('--model', type=str, help='Model file', required=True)
    parser.add_argument('--out', type=str, help='Wave output file', required=True)
    args = parser.parse_args(args=argv)

    print("Read data from: %s" % args.input, file=sys.stderr)
    with open(args.input) as f:
        data = f.read().rstrip("\n")
    print("Data len: %d" % len(data), file=sys.stderr)

    print("Model loading: %s" % args.model)
    device = torch.device("cpu")
    vocoder = load_model(args.model).to(device).eval()
    print("Model loaded - now ready to synthesize!")

    base64_bytes = data.encode('ascii')
    spectogram_bytes = base64.b64decode(base64_bytes)

    with torch.no_grad():
        start = time.time()
        x = torch.load(io.BytesIO(spectogram_bytes), map_location=device)
        y = vocoder.inference(x)
        # print(f"x = {x}")
        voc_end = time.time()
        elapsed = (voc_end - start)
        print(f"vocoder done: {elapsed:5f} s")
    audio_len = len(y) / sampling_rate
    rtf = elapsed / audio_len
    print(f"RTF = {rtf:5f}")
    print(f"Len = {audio_len:5f}")

    print("Write to: %s" % args.out, file=sys.stderr)
    with open(args.out, "wb") as wav_file:
        soundfile.write(wav_file, y.cpu().numpy(), sampling_rate, "PCM_16")

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

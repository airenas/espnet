#!/usr/bin/env python3
import argparse
import base64
import io
import sys
import time

import parallel_wavegan.models
import torch
import yaml
import soundfile


def main(argv):
    print("Starting", file=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Spectrogram input file', required=True)
    parser.add_argument('--model_dir', type=str, help='Model dir', required=True)
    parser.add_argument('--out', type=str, help='Wave output file', required=True)
    args = parser.parse_args(args=argv)

    print("Read data from: %s" % args.input, file=sys.stderr)
    with open(args.input) as f:
        data = f.read().rstrip("\n")
    print("Data len: %d" % len(data), file=sys.stderr)

    device = torch.device("cpu")
    vocoder_path = args.model_dir + "/checkpoint-700000steps.pkl"
    vocoder_conf = args.model_dir + "/config.yml"
    print("Model: %s" % vocoder_path, file=sys.stderr)
    with open(vocoder_conf) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
    vocoder = getattr(parallel_wavegan.models, vocoder_class)(**config["generator_params"])
    vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    pad_fn = torch.nn.ReplicationPad1d(config["generator_params"].get("aux_context_window", 0))
    print("Model loaded - now ready to synthesize!")

    base64_bytes = data.encode('ascii')
    spectogram_bytes = base64.b64decode(base64_bytes)

    with torch.no_grad():
        start = time.time()
        x = torch.load(io.BytesIO(spectogram_bytes))
        # print(f"x = {x}")
        c = pad_fn(x.unsqueeze(0).transpose(2, 1)).to(device)
        xx = (c,)
        z_size = (1, 1, (c.size(2) - sum(pad_fn.padding)) * config["hop_size"])
        z = torch.randn(z_size).to(device)
        xx = (z,) + xx
        y = vocoder(*xx).view(-1)
        voc_end = time.time()
        elapsed = (voc_end - start)
        print(f"vocoder done: {elapsed:5f} s")
    audio_len = len(y) / config["sampling_rate"]
    rtf = elapsed / audio_len
    print(f"RTF = {rtf:5f}")
    print(f"Len = {audio_len:5f}")

    print("Write to: %s" % args.out, file=sys.stderr)
    with open(args.out, "wb") as wav_file:
        soundfile.write(wav_file, y.cpu().numpy(), config["sampling_rate"], "PCM_16")

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

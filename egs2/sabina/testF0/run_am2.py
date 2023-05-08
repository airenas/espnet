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
    parser.add_argument('--model', type=str, help='Model dir', required=True)
    parser.add_argument('--model2', type=str, help='Model2 dir for pitch', required=True)
    parser.add_argument('--out', type=str, help='Spectrogram output file', required=True)
    args = parser.parse_args(args=argv)

    print("Read data from: %s" % args.data, file=sys.stderr)
    with open(args.data) as f:
        data = f.read().rstrip("\n")
    print("Data: %s" % data, file=sys.stderr)

    model_dir = extract_model(args.model)
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
    model2_dir = extract_model(args.model2)
    tts2 = Text2Speech(**model2_dir,
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

    # def pitch(d_masks):
    #     text = tts2.preprocess_fn("<dummy>", {"text": data})["text"]
    #     # batch = {"text": text}
    #     batch = to_device(text, tts2.device)
    #     x = F.pad(batch, [0, 1], "constant", tts2.tts.eos)
    #
    #     # setup batch axis
    #     ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
    #     xs, ys = x.unsqueeze(0), None
    #
    #     x_masks = make_non_pad_mask(ilens).to(x.device)
    #     x_masks = x_masks.unsqueeze(-2)
    #     hs, _ = tts2.tts.encoder(xs, x_masks)  # (B, Tmax, adim)
    #     p_outs = tts2.tts.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
    #     return p_outs

    # tts.model.tts.other_pitch = pitch
    tts.vocoder = None
    tts2.vocoder = None
    print("Model loaded - now ready to synthesize!")
    with torch.no_grad():
        start = time.time()
        res_v = tts2(text=data)

        def pitch(x, y):
            return res_v["pitch"].unsqueeze(0)

        del tts.model.tts.pitch_predictor
        tts.model.tts.pitch_predictor = pitch

        def energy(x, y):
            return res_v["energy"].unsqueeze(0)

        del tts.model.tts.energy_predictor
        tts.model.tts.energy_predictor = energy

        dur = InfFaker(data=res_v["duration"].unsqueeze(0))

        del tts.model.tts.duration_predictor
        tts.model.tts.duration_predictor = dur

        res = tts(text=data, decode_conf={"pitch": res_v["pitch"].unsqueeze(0)})
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


class InfFaker:
    def __init__(self, data):
        self.data = data

    def inference(self, x, y):
        return self.data


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import os
import sys
import time

import numpy as np
import torch
from parallel_wavegan.utils import load_model

from espnet2.bin.tts_inference import Text2Speech

fs = 22050


def to_int16(data):
    i = np.iinfo(np.int16)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (data * abs_max + offset).clip(i.min, i.max).astype(np.int16)


def write_wav(name, data):
    from scipy.io.wavfile import write
    write(name, fs, to_int16(data))


def rtf(start, end, w_len):
    return (end - start) / (w_len / fs)


def log_time(f, p1, p2):
    start_t = time.time()
    res = f(p1, p2)
    end_t = time.time()
    print(f"elapsed = {(end_t - start_t):5f}s")
    return res


def loadAM(amFile, dev):
    am_dir = os.path.dirname(amFile)
    mf = {'train_config': os.path.join(am_dir, 'config-run.yaml'),
          'model_file': amFile}
    text2speech = Text2Speech(
        **mf,
        device=dev,
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2
        speed_control_alpha=1.0,
    )
    text2speech.spc2wav = None  # Disable griffin-lim
    return text2speech


def loadVocoder(vocFile, dev):
    vocoder = load_model(vocFile).to(dev).eval()
    vocoder.remove_weight_norm()
    return vocoder


class cfg:
    def __init__(self, take_f0, take_energy, take_duration, interpolate_f0, f0_scale=1):
        self.take_f0 = take_f0
        self.take_energy = take_energy
        self.take_duration = take_duration
        self.interpolate_f0 = interpolate_f0
        self.f0_scale = f0_scale


class infFaker:
    def __init__(self, data):
        self.data = data

    def inference(self, x, y):
        return self.data

def get_next(start, data):
    for i in range(start + 1, len(data)):
        if data[i] >= -1.4:
            return data[i], i - start    
    return data[start], 0


def interpolate_f0(inp):
    data = inp
    pi = -1
    for i, x in enumerate(data):
        # print(x)
        if x < -1.4:
            v, c = get_next(i, data)
            if c > 0 and pi > -1:
                data[i] = data[pi] + (v - data[pi]) / (c + 1)
                pi = i
            elif pi == -1:
                data[i] = v
            elif c == 0:
                data[i] = v    
            print(f"F0 = {x} -> {data[i]}")
        else:
            pi = i    
    return data


def synthesize(phones, am, voc, am_f0, cfg: cfg):
    with torch.no_grad():
        start_am = time.time()
        if am_f0 is not None:
            am2_res = am_f0(phones)
            end_am2 = time.time()

            def pitch(x, y):
                res = am2_res["pitch"] 
                if cfg.interpolate_f0:
                    res = interpolate_f0(res)
                return (res * cfg.f0_scale).unsqueeze(0)

            def energy(x, y):
                return am2_res["energy"].unsqueeze(0)

            if cfg.take_f0:
                del am.model.tts.pitch_predictor
                am.model.tts.pitch_predictor = pitch
                print("Use pitch from am2")

            if cfg.take_energy:
                del am.model.tts.energy_predictor
                am.model.tts.energy_predictor = energy
                print("Use energy from am2")

            if cfg.take_duration:
                dur = infFaker(data=am2_res["duration"].unsqueeze(0))
                del am.model.tts.duration_predictor
                am.model.tts.duration_predictor = dur
                print("Use duration from am2")

        am_res = am(phones)
        end_am = time.time()
        wav = voc.inference(am_res["feat_gen"])
        end_voc = time.time()
    print(f"RTF all = {rtf(start_am, end_voc, len(wav)):5f}, time = {(end_voc - start_am):5f}s")
    if am_f0 is not None:
        print(f"RTF am2 = {rtf(start_am, end_am2, len(wav)):5f}, time = {(end_am2 - start_am):5f}s")
        print(f"RTF am  = {rtf(end_am2, end_am, len(wav)):5f}, time = {(end_am - end_am2):5f}s")
    else:
        print(f"RTF am  = {rtf(start_am, end_am, len(wav)):5f}, time = {(end_am - start_am):5f}s")
    print(f"RTF voc = {rtf(end_am, end_voc, len(wav)):5f}, time = {(end_voc - end_am):5f}s")
    return wav.view(-1).cpu().numpy(), am_res["pitch"].view(-1).cpu().numpy(), am_res["duration"].view(-1).cpu().numpy()


def main(argv):
    parser = argparse.ArgumentParser(description="Synthesizes wav file from phones",
                                     epilog="E.g. cat input.txt | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default='', type=str, help="Output File", required=True)
    parser.add_argument("--out-f0", default='', type=str, help="Output f0 File", required=True)
    parser.add_argument("--am", default='', type=str, help="AM File", required=True)
    parser.add_argument("--am2-f0", default='', type=str, help="AM File for f0", required=False)
    parser.add_argument("--am2-f0-scale", default=1, type=float, help="AM f0 scale", required=False)
    parser.add_argument("--am2-energy", default=False, help="Take energy from AM File for f0",
                        required=False, action='store_true')
    parser.add_argument("--am2-duration", default=False, help="Take duration from AM File for f0",
                        required=False, action='store_true')
    parser.add_argument("--am2-f0-interpolate", default=False, help="Do interpolate zero f0",
                        required=False, action='store_true')
    parser.add_argument("--voc", default='', type=str, help="Vocoder File", required=True)
    parser.add_argument("--dev", default='cpu', type=str, help="Device: cpu | cuda | cuda:1", required=False)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)
    lines = []
    for line in sys.stdin:
        s_line = line.strip()
        lines.append(s_line)
    phones = " ".join(lines).strip()

    print("Phones: == %s ==" % phones, file=sys.stderr)
    print("Loading AM from : %s" % args.am, file=sys.stderr)
    am = log_time(loadAM, args.am, args.dev)
    am_f0 = None
    if args.am2_f0:
        print("Loading AM2 from : %s" % args.am2_f0, file=sys.stderr)
        am_f0 = log_time(loadAM, args.am2_f0, args.dev)
    print("Loading Vocoder from : %s" % args.voc, file=sys.stderr)
    voc = log_time(loadVocoder, args.voc, args.dev)
    print("Synthesizing...", file=sys.stderr)
    data, f0, dur = synthesize(phones, am, voc, am_f0,
                               cfg=cfg(take_f0=am_f0 is not None, take_energy=args.am2_energy, 
                                       take_duration=args.am2_duration, interpolate_f0=args.am2_f0_interpolate, 
                                       f0_scale=args.am2_f0_scale))
    print("Saving audio", file=sys.stderr)
    if args.out_f0:
        np.savetxt(args.out_f0, f0, delimiter=',', fmt="%.5f")
        np.savetxt(args.out_f0 + ".dur", dur, delimiter=',', fmt="%d")
    write_wav(args.out, data)

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import logging
import os
import soundfile as sf

from datasets import tqdm


def main(args):
    wav_dir = args.dir
    if not os.path.isdir(wav_dir):
        logging.error(f"Dir {wav_dir} does not exist.")
        return

    wav_files = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]

    durations = []

    for filename in tqdm(wav_files, desc="Processing WAVs"):
        path = os.path.join(wav_dir, filename)
        try:
            data, samplerate = sf.read(path)
            duration = len(data) / samplerate
            durations.append(duration)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    if not durations:
        print("No WAV files processed.")
        exit()

    total_duration = sum(durations)
    avg_duration = total_duration / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)

    print("\n=== WAV Duration Statistics ===")
    print(f"Bendra trukmė             {total_duration:.2f} sek.")
    print(f"Vidutinė įrašo trukmė     {avg_duration:.2f} sek.")
    print(f"Min. įrašo trukmė         {min_duration:.2f} sek.")
    print(f"Maks. įrašo trukmė        {max_duration:.2f} sek.")
    print(f"Įrašų skaičius            {len(wav_files)}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Dir of audio files", )
    args = parser.parse_args()

    logging.info(f"Starting")

    main(args)

    logging.info("Done")

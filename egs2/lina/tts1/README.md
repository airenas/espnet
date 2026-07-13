# Lina's Text-to-Speech recipe

Written by Airenas Vaičiūnas @ VMU, Kaunas (2021)

## tts1 recipe

This is the recipe of Lithuanian single speaker TTS corpus.


### Prepare

#### Config

Make `Makefile.options.<speaker>` file with added info about corpus and output dirs:
```make
# path to corpus dir
corpus_dir?=corpus/ner/v01
# mlf file in corpus
corpus_mlf?=metadata.mlf
# utterance prefix in mlf
lab_prefix?=sint_i/NER/
corpus_wav_dir?=$(corpus_dir)/WAV44_match

# experiments dir
work_dir?=.data/sing/ner/v01
corpus_out_dir?=$(work_dir)/corpus
corpus?=ner
# speaker name
speaker?=NER
# speaker f0 ranges
f0min?=128
f0max=350

exclude_files?=
freq?=22050
step?=256   
```


### Train

```bash
make cfg=Makefile.options.<speaker> build
## or
nohup make cfg=Makefile.options.<speaker> build &
```

It will train and pack the model in working directory.

### Test

Sample hoe to use the packed model is provided in the [jupyter notebook](tts_demo.ipynb):

---



See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


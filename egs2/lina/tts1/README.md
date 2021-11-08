# Lina's Text-to-Speech recipe

Written by Airenas Vaičiūnas @ VMU, Kaunas (2021)

## tts1 recipe

This is the recipe of Lithuanian single female speaker TTS corpus.

### Train

```bash
make train-fs2
## or
nohup make train-fs2 > v01.log &
```

### Pack model

```bash
make pack-fastspeech2 inference_model=1500epoch.pth
```


See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


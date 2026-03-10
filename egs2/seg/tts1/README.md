# SEG (Sintezės emocinis garsynas) skriptai

This is the recipe of English single female speaker TTS model with [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


## Pasiruošimas

- Parsisiųskite kalbėtojo garsyną (pvz.: `AGN-1.0.zip`) 
- Sukurkite nustatymų failą, pvz. `Makefile.options`: 
```make
### garsyno failas
corpus_file?=corpus/AGN-1.0-sample.zip

### kalbėtojo duomenys
### pagrindinio tono rėžiai
f0min?=65
f0max?=330

### darbo katalogas, kuriame bus saugomi mokymo duomenys, eksperimentai, paruošti modeliai
work_dir?=agn-01
```

## Paruošiame duomenis mokinimui
```bash
make cfg=Makefile.options prepare/data
```



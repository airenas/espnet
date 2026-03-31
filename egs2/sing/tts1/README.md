# SINtezės Garsynas scripts

This is the recipe of Lithuanian single speaker TTS model with SING corpus.

Link to the SING Corpus: *TBD*

Detailed info in Lithuanian [README.lt](README.lt.md).


### Preparation

1. Download corpus zip from: `...<pending>`.
2. Prepare Makefile configuration file: `Makefile.options`:
   Add:
   1. full path to corpus file
   2. speaker 
   3. speaker f0 ranges
   4. working dir for the experiment

Sample:
```make
corpus_file?=/home/user/dwn/corpus/AGN-1.0.zip
speaker?=agn
f0min?=150
f0max?=625
work_dir?=agn-01
```

### Test configuration
Run `make info`
Expected output:
```txt
f0min: 			150
f0max: 			625
corpus_file: 	AGN-1.0.zip
work_dir: 		agn-01
speaker: 		agn
dev_count: 		250
nvidia-smi: 		NVIDIA RTX 4000 Ada Generation, 20475 MiB, 580.126.09
cuda visible dev: 	
python: 			Python 3.12.12
torch: 			2.10.0+cu128
cuda in python: 	12.8
```
Check that the corpus file and exp dir are correct. 
Check that cuda in python displays a version.

### Run model training
```bash
make build
## or in background
nohup make build &
```

A model will be trained and packed at: `${work_dir}/...`.

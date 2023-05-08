SHELL=/bin/bash

data?=data1.txt
wd=.wd/cust
load_f0?=""
# am=vytautas.v06b.fastspeech2.1500-01.zip
am?=sabina.v04.fastspeech2.1500-01.zip

$(wd):
	mkdir -p $@

# $(wd)/out.am: $(data) | $(wd)
# 	LOAD_F0=$(load_f0) WD=$(wd) PYTHONPATH=/home/airenas/projects/espnet python ./run_am.py --data $(data) --model $(am) --out $@_
# 	mv $@_ $@

$(wd)/out.am: $(data) | $(wd)
	LOAD_F0=$(load_f0) WD=$(wd) PYTHONPATH=/home/airenas/projects/espnet python ./run_am2.py --data $(data) --model $(am) \
	    --model2 vytautas.v06b.fastspeech2.1500-01.zip --out $@_
	mv $@_ $@


$(wd)/out.wav: $(wd)/out.am
	. ~/miniconda3/etc/profile.d/conda.sh; conda activate pwgan-cuda; \
	python ./run_voc.py --input $(wd)/out.am --model sabina.style.v04-100000/checkpoint-1000000steps.pkl --out .tmp.wav
	mv .tmp.wav $@

$(wd)/out.mp4: $(wd)/out.wav
	ffmpeg -i $^ $@

build: $(wd)/out.mp4

new: 
	touch $(data)

clean:
	rm -rf $(wd)

.wd/.orig:  
	$(MAKE) new build wd=.wd/orig
	touch $@
.wd/.vyt:  
	$(MAKE) new build wd=.wd/vyt am=vytautas.v06b.fastspeech2.1500-01.zip
	touch $@
.wd/.sab-vyt: .wd/.vyt
	$(MAKE) new build wd=.wd/sab-vyt load_f0=.wd/vyt/f0.txt
	touch $@	
.wd/.plain: .wd/f0.plain.txt
	$(MAKE) new build wd=.wd/plain load_f0=.wd/f0.plain.txt
	touch $@
.wd/.inc: .wd/f0.inc.txt
	$(MAKE) new build wd=.wd/inc load_f0=.wd/f0.inc.txt
	touch $@	
.wd/.low: .wd/f0.low.txt
	$(MAKE) new build wd=.wd/low load_f0=.wd/f0.low.txt
	touch $@		
.wd/f0.plain.txt: .wd/.orig
	cat .wd/orig/f0.txt | python make_f0.py --type=first > $@_
	mv $@_ $@
.wd/f0.inc.txt: .wd/.orig
	cat .wd/orig/f0.txt | python make_f0.py --type=inc > $@_
	mv $@_ $@
.wd/f0.low.txt: .wd/.orig
	cat .wd/orig/f0.txt | python make_f0.py --type=low > $@_
	mv $@_ $@	

build/all: .wd/.orig .wd/.vyt .wd/.sab-vyt .wd/.plain .wd/.inc  .wd/.low
cfg?=Makefile.options
-include $(cfg)
#####################################################################################
work_dir?=v-01
corpus_dir?=corpus/data
tts_dir=$(HOME_DIR)/gfs/tts
exp_dir=$(tts_dir)/espnet/egs2/lina/tts1/$(work_dir)
trans_type?=phn
exp?=tts_train_raw_phn_none
cuda?=0
ngpu?=1
test_count?=100
inference_model?=train.loss.ave.pth
speaker?=LIN
f0min?=128
f0max=350
# exp=char_train_no_dev_pytorch_train_pytorch_tacotron2
#####################################################################################
#### PREPARE CORPUS
#####################################################################################
#####################################################################################
#####################################################################################
python_local_cmd=PYTHONPATH=./../../../ LOGLEVEL=$(log) python3
		
${corpus_out_dir}/extracted ${corpus_out_dir}/lab ${corpus_out_dir}/range ${corpus_out_dir}/wav ${corpus_out_dir}/text \
	${corpus_out_dir}/lists $(corpus_out_dir)/$(corpus) $(corpus_out_dir)/$(corpus)/wavs $(work_dir)/downloads/${corpus}/wavs $(work_dir)/downloads/${corpus}:
	mkdir -p $@
############################################
${corpus_out_dir}/extracted/utf8.mlf: ${corpus_dir}/$(corpus_mlf) | ${corpus_out_dir}/extracted
	iconv -f ISO-8859-13 -t UTF-8 $< > $@
############################################
${corpus_out_dir}/extracted/fixed.mlf: ${corpus_out_dir}/extracted/utf8.mlf | ${corpus_out_dir}/lab
	pv $^ | $(python_local_cmd) ./local/exclude_files.py --exclude "$(exclude_files)" \
		| $(python_local_cmd) ./local/fix_mlf.py \
		| $(python_local_cmd) ./local/change_name.py --prefix '$(speaker)_' --strip-prefix '$(lab_prefix)' > $@
############################################
${corpus_out_dir}/extracted/fixed.split.mlf: ${corpus_out_dir}/extracted/fixed.mlf ${corpus_out_dir}/range/.done
	pv ${corpus_out_dir}/extracted/fixed.mlf | $(python_local_cmd) ./local/mlf_by_range.py --rangeDir ${corpus_out_dir}/range > $@_
	mv $@_ $@
############################################
${corpus_out_dir}/lab/.done: ${corpus_out_dir}/extracted/fixed.mlf | ${corpus_out_dir}/lab
	pv $^ | $(python_local_cmd) ./local/split_mlf.py --outDir `dirname $@`
	touch $@
############################################
${corpus_out_dir}/range/.done: ${corpus_out_dir}/lab/.done | ${corpus_out_dir}/range
	ls -1 ${corpus_out_dir}/lab/*.lab | xargs -n1 -I {} -P 4 sh -c \
		'base=`basename {} .lab`; \
		cat {} | $(python_local_cmd) ./local/mlf_to_range.py --max-sil 400 --max-duration 20000 > ${corpus_out_dir}/range/$${base}.split.range'
	touch $@
############################################
${corpus_out_dir}/wav/split.sh: ${corpus_out_dir}/range/.done | ${corpus_out_dir}/wav
	rm -f $@_
	ls -1 ${corpus_out_dir}/range/*.range | xargs -n1 -I {} sh -c \
		'base=`basename {} .split.range`; \
		cat {} | $(python_local_cmd) ./local/range_to_split_sh.py --name $${base} --in-dir $(corpus_wav_dir) --out-dir ${corpus_out_dir}/wav \
			--strip-prefix '$(speaker)_' >> $@_'
	mv $@_ $@	
	chmod +x $@
############################################
${corpus_out_dir}/wav/.done: ${corpus_out_dir}/wav/split.sh
	${corpus_out_dir}/wav/split.sh
	touch ${corpus_out_dir}/wav/.done
############################################
${corpus_out_dir}/extracted/metadata.csv: ${corpus_out_dir}/extracted/fixed.split.mlf
	pv $< | $(python_local_cmd) ./local/mlf_to_csv.py --outputPhones --skipSP > $@_
	mv $@_ $@
$(work_dir)/downloads/${corpus}/metadata.csv: ${corpus_out_dir}/extracted/metadata.csv | $(work_dir)/downloads/${corpus}
	pv $< | sed '/noise/d' | sort > $@_
	mv $@_ $@
$(work_dir)/downloads/${corpus}/durations: ${corpus_out_dir}/extracted/fixed.split.mlf ${corpus_out_dir}/extracted/audio.samples \
	${corpus_out_dir}/extracted/exclude.files | $(work_dir)/downloads/${corpus}		
	pv $< | $(python_local_cmd) ./local/exclude_files.py --excludeFromFile ${corpus_out_dir}/extracted/exclude.files \
		| $(python_local_cmd) ./local/mlf_to_durations.py --shift $(step) --freq $(freq) --samplesFile ${corpus_out_dir}/extracted/audio.samples \
		| sort > $@_
	mv $@_ $@

############################################
${corpus_out_dir}/extracted/audio.samples: $(work_dir)/downloads/${corpus}/wavs/.done
	ls -1 $(work_dir)/downloads/${corpus}/wavs/*.wav | sort | pv | $(python_local_cmd) ./local/audio_to_samples.py > $@_
	mv $@_ $@
############################################
${corpus_out_dir}/extracted/exclude.files: ${corpus_out_dir}/extracted/metadata.csv
	cat $< | grep 'noise' | awk -F'|' '{print $$1}' > $@_
	mv $@_ $@
############################################
$(work_dir)/downloads/${corpus}/wavs/.done: ${corpus_out_dir}/wav/.done ${corpus_out_dir}/extracted/exclude.files | $(work_dir)/downloads/${corpus}/wavs
	for wav in ${corpus_out_dir}/wav/*.wav; do \
		base=`basename $${wav} .wav`; \
		base=$$(echo $${base} | sed s/lab_//g); \
# 		echo "Copy wav $${base}"; \
# 		cp $${wav} $(work_dir)/downloads/${corpus}/wavs/$${base}.wav; \
		echo "Converting WAV to $(freq) Hz 16 bit for $${base}"; \
		sox $${wav} -r $(freq) -b 16 $(work_dir)/downloads/${corpus}/wavs/$${base}.wav; \
	done
	echo "Drop exluded files"
	cat ${corpus_out_dir}/extracted/exclude.files | xargs -n 1 -I {} sh -c 'rm $(work_dir)/downloads/${corpus}/wavs/{}.wav'
	touch $(work_dir)/downloads/${corpus}/wavs/.done
############################################
test/durations: $(work_dir)/downloads/${corpus}/durations
	cat $< | $(python_local_cmd) ./local/inspect_durations.py --duration 150 --shift $(step) --freq $(freq)
test/corpus/len: $(work_dir)/downloads/${corpus}/wavs/.done
	ls -1 $(work_dir)/downloads/${corpus}/wavs/*.wav | xargs -n1 soxi -D | $(python_local_cmd) -c "import sys; print(f'{sum(float(x) for x in sys.stdin)/3600:.3f}h')"

$(bin):
	mkdir -p $(bin)
############################################
build/meta: $(corpus_out_dir)/$(corpus)/metadata.csv
$(work_dir)/.corpus.prepare.done: $(work_dir)/downloads/${corpus}/metadata.csv $(work_dir)/downloads/${corpus}/durations
	touch $@
prepare/corpus: $(work_dir)/.corpus.prepare.done
############################################
clean/corpus:
	@echo -n "Are you sure to delete $(corpus_out_dir)? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(corpus_out_dir)
#####################################################################################
#####################################################################################
#####################################################################################
train/all: $(work_dir)/.fs2.eval.done

prepare/initial: $(work_dir)/.initial.prepare.done

$(work_dir)/.initial.prepare.done: $(work_dir)/.corpus.prepare.done
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 0 --stop-stage 4 --trans_type $(trans_type)
	touch $@

fastspeech2_params=--trans_type $(trans_type) \
		--train_config conf/tuning/train_fastspeech2.yaml \
    	--teacher_dumpdir $(work_dir)/duration \
    	--tts_stats_dir $(work_dir)/exp/tts_train_raw_phn_none/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    	--write_collected_feats true

$(work_dir)/duration/tr_no_dev:
	mkdir -p $@
$(work_dir)/duration/dev:
	mkdir -p $@
$(work_dir)/duration/eval1:
	mkdir -p $@		

$(work_dir)/duration/tr_no_dev/durations: $(work_dir)/.initial.prepare.done | $(work_dir)/duration/tr_no_dev
	cat $(work_dir)/dump/raw/tr_no_dev/text | cut -d ' ' -f1 > $@.files
	grep -Ff $@.files $(work_dir)/downloads/$(corpus)/durations > $@

$(work_dir)/duration/dev/durations: $(work_dir)/.initial.prepare.done | $(work_dir)/duration/dev
	cat $(work_dir)/dump/raw/dev/text | cut -d ' ' -f1 > $@.files
	grep -Ff $@.files $(work_dir)/downloads/$(corpus)/durations > $@	

$(work_dir)/duration/eval1/durations: $(work_dir)/.initial.prepare.done | $(work_dir)/duration/eval1
	cat $(work_dir)/dump/raw/eval1/text | cut -d ' ' -f1 > $@.files
	grep -Ff $@.files $(work_dir)/downloads/$(corpus)/durations > $@

$(work_dir)/.fs2.prepare.done: $(work_dir)/duration/tr_no_dev/durations $(work_dir)/duration/dev/durations $(work_dir)/duration/eval1/durations
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 5 --stop-stage 5 $(fastspeech2_params)
	touch $@

$(work_dir)/.fs2.train.done: $(work_dir)/.fs2.prepare.done
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 6 --stop-stage 6 $(fastspeech2_params)
	touch $@

$(work_dir)/.fs2.eval.done: $(work_dir)/.fs2.train.done
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 7 --stop-stage 7 $(fastspeech2_params)
	touch $@

prepare-fs2: $(work_dir)/.fs2.prepare.done 
train-fs2: $(work_dir)/.fs2.train.done
eval-fs2: $(work_dir)/.fs2.eval.done
pack-fastspeech2: 
	./run.sh --stage 8 --stop-stage 8 --inference_model ${inference_model} $(fastspeech2_params)

train-one:
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage $(stage) --stop-stage $(stage) --trans_type $(trans_type)
#####################################################################################
put-data:
	$(SSH_CMD) "mkdir -p $(exp_dir)/downloads"
	scp -P $(SSH_PORT) $(corpus_file) $(USER)@$(SERVER_URL):$(exp_dir)/downloads
#####################################################################################
clean:
	@echo -n "Are you sure to delete $(work_dir)/...? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/data $(work_dir)/exp $(work_dir)/dump $(work_dir)/exp $(work_dir)/fbank \
		$(work_dir)/tensorboard $(work_dir)/duration
	rm -f $(work_dir)/.*.*.*
clean/data:
	@echo -n "Are you sure to delete $(work_dir)/downloads? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/downloads
#####################################################################################
.EXPORT_ALL_VARIABLES:	

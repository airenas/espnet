cfg?=Makefile.options
-include $(cfg)
#####################################################################################
work_dir?=v-01
exp_dir=$(tts_dir)/espnet/egs2/laimis/tts1/$(work_dir)
trans_type?=phn
exp?=tts_train_raw_phn_none
cuda?=0
ngpu?=1
test_count?=100
inference_model?=train.loss.ave.pth
f0min?=80
f0max?=210
# exp=char_train_no_dev_pytorch_train_pytorch_tacotron2
#####################################################################################
$(work_dir)/downloads $(work_dir):
	mkdir -p $@
#####################################################################################
# DATA PREPARATION
#####################################################################################
info: 
	@echo "f0min: $(f0min)"
	@echo "f0max: $(f0max)"
	@echo "seg_corpus: $(seg_corpus)"
	@echo "work_dir: $(work_dir)"

.PHONY: info

prepare/corpus: info $(work_dir)/.corpus.done 
.PHONY: prepare/corpus

$(work_dir)/.corpus.done: | $(work_dir)
	./run.sh --stage -1 --stop-stage 1
	touch $@	

prepare/data: $(work_dir)/.initial.prepare.done

$(work_dir)/.initial.prepare.done: $(work_dir)/.corpus.done
	./run.sh --stage 1 --stop-stage 4
	touch $@
######################################################################################
# TRAINING
######################################################################################

train-all: $(work_dir)/.fs2.eval.done


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
	@echo -n "Are you sure you want to delete $(work_dir)/...? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/data $(work_dir)/exp $(work_dir)/dump $(work_dir)/exp $(work_dir)/fbank \
		$(work_dir)/tensorboard $(work_dir)/duration
	rm -f $(work_dir)/.*.done
.PHONY: clean	
clean/data:
	@echo -n "Are you sure you want to delete $(work_dir)/downloads? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/downloads
.PHONY: clean/data	
#####################################################################################
.EXPORT_ALL_VARIABLES:	

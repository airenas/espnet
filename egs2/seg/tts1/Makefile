cfg?=Makefile.options
-include $(cfg)
#####################################################################################
work_dir?=v-01
trans_type?=phn
## of tacotron2
inference_model?=train.loss.ave
## for packing fastspeech2
final_fs2_model?=train.loss.ave
## speakers specific parameters
f0min?=70
f0max?=500
## how many samples to use for dev set
dev_count?=250
#####################################################################################
$(work_dir)/downloads $(work_dir):
	mkdir -p $@
#####################################################################################
info: 
	@echo "f0min: 			$(f0min)"
	@echo "f0max: 			$(f0max)"
	@echo "corpus_file: 	$(corpus_file)"
	@echo "work_dir: 		$(work_dir)"
	@echo "speaker: 		${speaker}"
	@echo "dev_count: 		${dev_count}"
	@echo "nvidia-smi: 		$$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tail -n1)"
	@echo "cuda visible dev: 	$$(echo $$CUDA_VISIBLE_DEVICES)"
	@echo "python: 			$$(python --version)"
	@echo "torch: 			$$(python -c 'import torch; print(torch.__version__)')"
	@echo "cuda in python: 	$$(python -c 'import torch; print(torch.version.cuda) if torch.cuda.is_available() else "N/A"')"
.PHONY: info
######################################################################################
#####################################################################################
# DATA PREPARATION
#####################################################################################
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
tacotron_params=--trans_type $(trans_type) \
		--train_config conf/tuning/train_tacotron2.yaml
$(work_dir)/.tacotron.done: $(work_dir)/.initial.prepare.done
	./run.sh --stage 5 --stop-stage 7 $(tacotron_params)
	touch $@
train/tacotron: $(work_dir)/.tacotron.done
.PHONY: train/tacotron
######################################################################################
# GET ALIGNMENTS
######################################################################################
tacotron_decode_params=--trans_type $(trans_type) \
		--train_config conf/tuning/train_tacotron2.yaml \
		--inference_config conf/decode.yaml \
		--inference_model ${inference_model}.pth \
		--inference_args "--use_teacher_forcing true" --test_sets "tr_no_dev dev eval1"
$(work_dir)/.tacotron.decode.done: $(work_dir)/.tacotron.done
	./run.sh --stage 8 --stop-stage 8 $(tacotron_decode_params)
	touch $@
decode/tacotron: $(work_dir)/.tacotron.decode.done
.PHONY: decode/tacotron
######################################################################################
# FINAL FASTSPEECH2 TRAINING
######################################################################################
fs2_params=--trans_type $(trans_type) \
		--train_config conf/tuning/train_fastspeech2.yaml \
    	--teacher_dumpdir $(work_dir)/exp/tts_train_tacotron2_raw_phn_espeak_ng_lt/decode_use_teacher_forcingtrue_${inference_model} \
    	--tts_stats_dir $(work_dir)/exp/tts_train_tacotron2_raw_phn_espeak_ng_lt/decode_use_teacher_forcingtrue_${inference_model}/stats \
    	--write_collected_feats true
$(work_dir)/.fs2.done: $(work_dir)/.tacotron.decode.done
	./run.sh --stage 5 --stop-stage 7 $(fs2_params)
	touch $@
train/fs2: $(work_dir)/.fs2.done
.PHONY: train/fs2
######################################################################################
# PACK MODEL
######################################################################################
fs2_pack_params=--trans_type $(trans_type) \
		--train_config conf/tuning/train_fastspeech2.yaml \
		--skip_packing false \
		--inference_model ${final_fs2_model}.pth 
pack/fs2: $(work_dir)/.fs2.done
	./run.sh --stage 10 --stop-stage 10 $(fs2_pack_params)
	# add missing files
	zip -r $(work_dir)exp/tts_train_fastspeech2_raw_phn_espeak_ng_lt/tts_train_fastspeech2_raw_phn_espeak_ng_lt_${final_fs2_model}.zip \
		$(work_dir)/exp/tts_train_tacotron2_raw_phn_espeak_ng_lt/decode_use_teacher_forcingtrue_${inference_model}/stats/train/*.npz
.PHONY: pack/fs2
#####################################################################################		
build: pack/fs2
.PHONY: build
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

-include Makefile.options
#####################################################################################
USER=avaiciunas
SERVER_URL?=agpu
SSH_PORT?=22
HOME_DIR?=/home/avaiciunas

SSH_CMD=ssh -p $(SSH_PORT) -t $(USER)@$(SERVER_URL)
SSH_TF_CMD=ssh -L 6006:localhost:6006 -p $(SSH_PORT) -t $(USER)@$(SERVER_URL)
SSH_JN_CMD=ssh -L 8888:localhost:8888 -p $(SSH_PORT) -t $(USER)@$(SERVER_URL)
#####################################################################################
work_dir?=g-01
tts_dir=$(HOME_DIR)/gfs/tts
exp_dir=$(tts_dir)/espnet/egs2/sabina/tts1/$(work_dir)
trans_type?=phn
exp?=tts_train_raw_phn_none
cuda?=0
ngpu?=1
test_count?=100
inference_model?=train.loss.ave.pth
# exp=char_train_no_dev_pytorch_train_pytorch_tacotron2
#####################################################################################
$(work_dir)/downloads:
	mkdir -p $@

init-data: | $(work_dir)/downloads	
#####################################################################################
logs:
	$(SSH_CMD) "cd $(exp_dir) && tail -n 40 -f exp/$(exp)/train.log"

nvidia:
	$(SSH_CMD) "watch -n 10 nvidia-smi"
#####################################################################################
tensorboard:
	$(SSH_TF_CMD) "source ~/miniconda3/etc/profile.d/conda.sh;conda activate tts; \
		cd $(exp_dir)/exp/$(exp); pwd; \
		tensorboard --logdir tensorboard/"

jupyter:
	$(SSH_JN_CMD) "source ~/miniconda3/etc/profile.d/conda.sh;conda activate tts; \
		cd $(tts_dir); pwd; jupyter notebook"
#####################################################################################
ssh:
	$(SSH_CMD)
#####################################################################################
train-all: $(work_dir)/.fs2.eval.done

prepare-tracotron2: $(work_dir)/.trac2.prepare.done
train-tracotron2: $(work_dir)/.trac2.train.done
eval-tracotron2: $(work_dir)/.trac2.eval.done
$(work_dir)/.trac2.train.done: $(work_dir)/.trac2.prepare.done
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 6 --stop-stage 6 --trans_type $(trans_type)
	touch $@

$(work_dir)/.trac2.prepare.done:
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 0 --stop-stage 5 --trans_type $(trans_type)
	touch $@

$(work_dir)/.trac2.eval.done: $(work_dir)/.trac2.train.done
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 7 --stop-stage 7 --trans_type $(trans_type) \
		--tts_exp $(work_dir)/exp/tts_train_raw_phn_none \
	    --inference_args "--use_teacher_forcing true" \
	    --test_sets "tr_no_dev dev eval1"
	touch $@

pack-tracotron2: 
	./run.sh --stage 8 --stop-stage 8

fastspeech2_params=--trans_type $(trans_type) \
		--train_config conf/tuning/train_fastspeech2.yaml \
    	--teacher_dumpdir $(work_dir)/exp/tts_train_raw_phn_none/decode_use_teacher_forcingtrue_train.loss.ave \
    	--tts_stats_dir $(work_dir)/exp/tts_train_raw_phn_none/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    	--write_collected_feats true

$(work_dir)/.fs2.prepare.done: $(work_dir)/.trac2.eval.done
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
	./run.sh --stage 8 --stop-stage 8 $(fastspeech2_params)

train-one:
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage $(stage) --stop-stage $(stage) --trans_type $(trans_type)
#####################################################################################
put-data:
	$(SSH_CMD) "mkdir -p $(exp_dir)/downloads"
	scp -P $(SSH_PORT) $(corpus_file) $(USER)@$(SERVER_URL):$(exp_dir)/downloads
#####################################################################################
clean:
	@echo -n "Are you sure to delete $(work_dir)/...? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/data $(work_dir)/exp $(work_dir)/dump $(work_dir)/exp $(work_dir)/fbank $(work_dir)/tensorboard
	rm -f $(work_dir)/.*

clean-data:
	@echo -n "Are you sure to delete $(work_dir)/downloads? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/downloads
#####################################################################################
.EXPORT_ALL_VARIABLES:	

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
exp_dir=$(tts_dir)/espnet/egs/sabina/tts1/$(work_dir)
trans_type?=char
exp?=$(trans_type)_train_no_dev_pytorch_train_pytorch_tacotron2
cuda?=0
# exp=char_train_no_dev_pytorch_train_pytorch_tacotron2
#####################################################################################
$(work_dir)/downloads:
	mkdir -p $@

init-data: | $(work_dir)/downloads	
#####################################################################################
logs:
	$(SSH_CMD) "cd $(exp_dir) && tail -f exp/$(exp)/train.log"

nvidia:
	$(SSH_CMD) "watch nvidia-smi"
#####################################################################################
tensorboard:
	$(SSH_TF_CMD) "source ~/miniconda3/etc/profile.d/conda.sh;conda activate tts; \
		cd $(exp_dir); pwd; \
		tensorboard --logdir tensorboard/$(exp)/"	

jupyter:
	$(SSH_JN_CMD) "source ~/miniconda3/etc/profile.d/conda.sh;conda activate tts; \
		cd $(tts_dir); pwd; jupyter notebook"
#####################################################################################
ssh:
	$(SSH_CMD)
#####################################################################################
train: 
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage 3 --stop-stage 3 --trans_type $(trans_type)			

train-one: 
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage $(stage) --stop-stage $(stage) --trans_type $(trans_type)

train-many: 
	CUDA_VISIBLE_DEVICES=$(cuda) ./run.sh --stage $(from) --stop-stage $(to) --trans_type $(trans_type)

#####################################################################################
put-data:
	scp -P $(SSH_PORT) $(corpus_file) $(USER)@$(SERVER_URL):$(exp_dir)/downloads

#####################################################################################
clean:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/data $(work_dir)/exp $(work_dir)/dump $(work_dir)/exp $(work_dir)/fbank $(work_dir)/tensorboard

clean-data:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(work_dir)/downloads
#####################################################################################
.EXPORT_ALL_VARIABLES:	

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
tts_dir=$(HOME_DIR)/gfs/tts
exp_dir=$(tts_dir)/espnet/egs/sabina/tts1
exp=char_train_no_dev_pytorch_train_pytorch_transformer.v3.single
# exp=char_train_no_dev_pytorch_train_pytorch_tacotron2
#####################################################################################
logs:
	$(SSH_CMD) "cd $(exp_dir) && tail -f exp/$(exp)/train.log"

nvidia:
	$(SSH_CMD) "watch nvidia-smi"

tensorboard:
	$(SSH_TF_CMD) "source ~/miniconda3/etc/profile.d/conda.sh;conda activate tts; \
		cd $(exp_dir); pwd; \
		tensorboard --logdir tensorboard/$(exp)/"	

jupyter:
	$(SSH_JN_CMD) "source ~/miniconda3/etc/profile.d/conda.sh;conda activate tts; \
		cd $(tts_dir); pwd; jupyter notebook"	

-include $(cfg)
###########################################
work_dir?=.work
log ?= INFO
python_local_cmd?=PYTHONPATH=./../../../ LOGLEVEL=$(log) python3
sota_model_dir?=./.sota-model
###########################################
info:
	@echo "cfg      = ${cfg}"
	@echo "work_dir = ${work_dir}"
	@echo "am       = ${am}"
	@echo "am_file  = ${am_file}"
	@echo "vocoder  = ${vocoder}"
	@echo "vocoder_file = ${vocoder_file}"
	@echo "input    = ${input}"

.PHONY: info
###########################################
${work_dir}/am/model ${work_dir}/vocoder/model ${work_dir}/audio $(sota_model_dir):
	mkdir -p $@
###########################################
${work_dir}/am/.dwn: | ${work_dir}/am/model
	mr fetch ${am} ${work_dir}/am/model --force
	touch $@
${work_dir}/vocoder/.dwn: | ${work_dir}/vocoder/model
	mr fetch ${vocoder} ${work_dir}/vocoder/model --force
	touch $@
###########################################
${work_dir}/audio/.done: ${work_dir}/vocoder/.dwn ${work_dir}/am/.dwn | ${work_dir}/audio
	${python_local_cmd} ./local/synthesize.py \
		--am ${work_dir}/am/model/$(am_file) --vocoder ${work_dir}/vocoder/model/$(vocoder_file) \
		--input ${input} --output_dir ${work_dir}/audio
	touch $@
###########################################
$(sota_model_dir)/.dwn: | $(sota_model_dir)
	hf download Scicom-intl/SOTA-MOS --include "model/onnx/*" --local-dir $(sota_model_dir)
	touch $@
$(sota_model_dir)/.uv.done: $(sota_model_dir)/.dwn 	
	uv sync
	touch $@
init/sota: $(sota_model_dir)/.uv.done
.PHONY: init/sota
###########################################
${work_dir}/prediction.csv: ${work_dir}/audio/.done $(sota_model_dir)/.dwn $(sota_model_dir)/.uv.done
	uv run python $(sota_model_dir)/model/onnx/onnx_predict.py --model $(sota_model_dir)/model/onnx \
		--out ${work_dir}/prediction.csv ${work_dir}/audio/*.wav 
###########################################
build: ${work_dir}/prediction.csv
	LOGLEVEL=$(log) uv run python ./local/eval.py --input ${work_dir}/prediction.csv
.PHONY: build
###########################################
clean:
	rm -rf $(work_dir)
	rm -rf $(sota_model_dir)
	rm -rf .venv .cache
.PHONY: clean
###########################################
-include $(cfg)
###########################################
work_dir?=.work
log ?= INFO
python_local_cmd?=PYTHONPATH=./../../../ LOGLEVEL=$(log) python3
uv_cmd?=LOGLEVEL=$(log) uv run python
sota_model_dir?=./.sota-model
###########################################
info:
	@echo "cfg      = ${cfg}"
	@echo "work_dir = ${work_dir}"
	@echo "corpus_s3_dir = ${corpus_s3_dir}"
	@echo "am       = ${am}"
	@echo "am_file  = ${am_file}"
	@echo "vocoder  = ${vocoder}"
	@echo "vocoder_file = ${vocoder_file}"
	@echo "input    = ${input}"

.PHONY: info
###########################################
${work_dir}/am/model ${work_dir}/vocoder/model ${work_dir}/audio $(sota_model_dir) \
	${work_dir}/corpus ${work_dir}/audio-corpus:
	mkdir -p $@
###########################################
${work_dir}/am/.dwn: | ${work_dir}/am/model
	mr fetch ${am} ${work_dir}/am/model --force
	touch $@
${work_dir}/vocoder/.dwn: | ${work_dir}/vocoder/model
	mr fetch ${vocoder} ${work_dir}/vocoder/model --force
	touch $@
${work_dir}/corpus/.dwn: | ${work_dir}/corpus
	rc object copy -r $(corpus_s3_dir) ${work_dir}/corpus
	# leave only the first 200 files
	find ${work_dir}/corpus -type f | sort | tail -n +201 | xargs -r rm -f
	touch $@
###########################################
${work_dir}/audio/.done: ${work_dir}/vocoder/.dwn ${work_dir}/am/.dwn | ${work_dir}/audio
	${python_local_cmd} ./local/synthesize.py \
		--am ${work_dir}/am/model/$(am_file) --vocoder ${work_dir}/vocoder/model/$(vocoder_file) \
		--input ${input} --output_dir ${work_dir}/audio
	touch $@
###########################################
${work_dir}/audio-corpus/.done: ${work_dir}/corpus/.dwn | ${work_dir}/audio-corpus
	LOGLEVEL=$(log) uv run python ./local/prepare_audio.py --input ${work_dir}/corpus --output ${work_dir}/audio-corpus --segment-duration 20.0
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
${work_dir}/prediction_utmosv2.csv: ${work_dir}/audio/.done $(sota_model_dir)/.dwn $(sota_model_dir)/.uv.done
	$(uv_cmd) ./local/utmosv2_predict.py --out $@_ ${work_dir}/audio/*.wav
	mv $@_ $@
###########################################
${work_dir}/prediction_sotamos.csv: ${work_dir}/audio/.done $(sota_model_dir)/.dwn $(sota_model_dir)/.uv.done
	$(uv_cmd) $(sota_model_dir)/model/onnx/onnx_predict.py --model $(sota_model_dir)/model/onnx \
		--out ${work_dir}/prediction_sotamos.csv ${work_dir}/audio/*.wav
###########################################
${work_dir}/prediction_corpus.csv: ${work_dir}/audio-corpus/.done $(sota_model_dir)/.dwn $(sota_model_dir)/.uv.done
	$(uv_cmd) $(sota_model_dir)/model/onnx/onnx_predict.py --model $(sota_model_dir)/model/onnx \
		--out ${work_dir}/prediction_corpus.csv ${work_dir}/audio-corpus/*.wav 		
###########################################
build/sotamos: ${work_dir}/prediction_sotamos.csv
	$(uv_cmd) ./local/eval.py --input ${work_dir}/prediction_sotamos.csv
.PHONY: build/sotamos
###########################################
build/utmosv2: ${work_dir}/prediction_utmosv2.csv
	$(uv_cmd) ./local/eval.py --input ${work_dir}/prediction_utmosv2.csv
.PHONY: build/utmosv2
###########################################
build/corpus: ${work_dir}/prediction_corpus.csv
	$(uv_cmd) ./local/eval.py --input ${work_dir}/prediction_corpus.csv
.PHONY: build/corpus
###########################################
clean/all: clean
	rm -rf $(sota_model_dir)
	rm -rf .venv .cache
.PHONY: clean/all
clean:
	rm -rf $(work_dir)
.PHONY: clean
###########################################
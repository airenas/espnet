-include Makefile.options

out_dir=out
f0_in:=$(shell find f0-analysis/pitch/ -name '*.npy')
f0_files=$(patsubst f0-analysis/pitch/%,%,$(patsubst %.npy,%,$(f0_in)))
f0_res=$(patsubst %,$(out_dir)/%.f0.info,$(f0_files))
f0d_res=$(patsubst %,$(out_dir)/%.f0,$(f0_files))
f0p_res=$(patsubst %,$(out_dir)/%.f0.pdf,$(f0_files))

info:
	echo $(f0_res)

$(out_dir):
	mkdir -p $@
$(out_dir)/%.f0.info: f0-analysis/pitch/%.npy | $(out_dir)
	python restore_f0.py --input $^ --txts f0-analysis/srctexts --name $* --durs f0-analysis/durations > $@_
	mv $@_ $@
$(out_dir)/%.f0: $(out_dir)/%.f0.info | $(out_dir)
	python to_dur_f0.py --input $^ > $@_
	mv $@_ $@
$(out_dir)/%.f0.pdf: $(out_dir)/%.f0.info | $(out_dir)
	python plot_f0.py --input $^ --output $@

build: $(f0_res) $(f0d_res) $(f0p_res) | $(out_dir)
	
clean:
	rm -rf $(out_dir)

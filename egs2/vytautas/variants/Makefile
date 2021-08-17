-include Makefile.options

out_dir?=out
sentences?=s01 s02 s03 s04 s05
mp3_files=$(patsubst %,$(out_dir)/%_$(am)-$(amv)_$(voc)-$(vocv).mp3,$(sentences))
#####################################################################################
$(out_dir):
	mkdir -p $@
$(out_dir)/%.mp3: $(out_dir)/%.wav
	ffmpeg -i $< $@.mp3
	mv $@.mp3 $@
$(out_dir)/%_$(am)-$(amv)_$(voc)-$(vocv).wav: $(am-$(am)-dir)/$(amv).pth $(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl | $(out_dir)
	cat sentences/$*.txt | python synthesize.py --out $@_ --am $(am-$(am)-dir)/$(amv).pth \
		--voc=$(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl
	mv $@_ $@
#####################################################################################
am-%:
	$(MAKE) am=$* $(am-$*-vrs)
amv-%:
	$(MAKE) amv=$* $(vocs)
voc-%:
	$(MAKE) voc=$* $(voc-$*-vrs)
vocv-%:
	$(MAKE) vocv=$* build
#####################################################################################
build: $(mp3_files)
	echo $(mp3_files)
build-all:
	$(MAKE) $(ams)

#####################################################################################
mp3s:
	mkdir -p $@

dwn-mp3: | mp3s
	rsync -r -P avaiciunas@agpu:/home/avaiciunas/gfs/tts/espnet/egs2/vytautas/variants/out/*.mp3 mp3s

out.html: mp3s display_html.py
	ls -1 mp3s/*.mp3 | python display_html.py > $@_
	mv $@_ $@

generate_html: out.html

clean:
	rm -f out.html


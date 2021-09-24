-include Makefile.options

out_dir?=out
sentences?=s01 s02 s03 s04 s05 s06 s07 s08 s09
mp3_files=$(patsubst %,$(out_dir)/%_$(am)-$(amv)_$(voc)-$(vocv).mp3,$(sentences))
dev?=cpu
#####################################################################################
$(out_dir):
	mkdir -p $@
$(out_dir)/%.mp3: $(out_dir)/%.wav
	ffmpeg -y -i $< $@.mp3
	mv $@.mp3 $@
$(out_dir)/%_$(am)-$(amv)_$(voc)-$(vocv).wav: $(am-$(am)-dir)/$(amv).pth $(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl | $(out_dir)
ifneq ($(dev), cpu)
	flock .cuda.lock -c "cat sentences/$*.txt | python synthesize.py --out $@_ --am $(am-$(am)-dir)/$(amv).pth \
		--voc=$(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl --dev $(dev)"
else
	cat sentences/$*.txt | python synthesize.py --out $@_ --am $(am-$(am)-dir)/$(amv).pth \
		--voc=$(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl --dev $(dev)
endif
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
	rsync -r -P avaiciunas@agpu:/home/avaiciunas/gfs/tts/espnet/egs2/laimis/variants/out/*.mp3 mp3s

gt_files=$(patsubst %,mp3s/%_gt.mp3,$(sentences))
make-gt: $(gt_files)
mp3s/s08_gt.mp3: | mp3s
	ffmpeg -i $(corpus)/lab_LAI_CIZ_A030_000.wav $@	
mp3s/s09_gt.mp3: | mp3s
	ffmpeg -i $(corpus)/lab_LAI_CIZ_A030_001.wav $@

put-online:
	rsync -P out.html airenas@sinteze.intelektika.lt:~/intelektikaLt-deploy/deploy/c8/tts/html/laimis-samples/
	rsync -rP mp3s airenas@sinteze.intelektika.lt:~/intelektikaLt-deploy/deploy/c8/tts/html/laimis-samples/


out.html: mp3s display_html.py
	ls -1 mp3s/*_*_*.mp3 | python display_html.py > $@_
	mv $@_ $@

generate_html: out.html

clean:
	rm -f out.html


-include Makefile.options

out_dir?=out
sentences?=s01 s02 s03 s04 s05 s06 s07
mp3_files=$(patsubst %,$(out_dir)/%_$(am)-$(amv)_$(voc)-$(vocv).mp3,$(sentences))
dev?=cpu
maxcuda?=2
#####################################################################################
$(out_dir):
	mkdir -p $@
$(out_dir)/%.mp3: $(out_dir)/%.wav
	ffmpeg -y -i $< $@.mp3
	mv $@.mp3 $@
$(out_dir)/%_$(am)-$(amv)_$(voc)-$(vocv).wav: $(am-$(am)-dir)/$(amv).pth $(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl | $(out_dir)
ifneq ($(dev), cpu)
ifeq ($(dev), cuda)
	a=$$(($$RANDOM%$(maxcuda))); flock .cuda.$$a.lock -c "cat sentences/$*.txt | python synthesize.py --out $@_ --am $(am-$(am)-dir)/$(amv).pth \
		--voc=$(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl --dev cuda:$$a"
else
	flock .cuda.lock -c "cat sentences/$*.txt | python synthesize.py --out $@_ --am $(am-$(am)-dir)/$(amv).pth \
		--voc=$(voc-$(voc)-dir)/checkpoint-$(vocv)steps.pkl --dev $(dev)"		
endif		
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
	rsync -r -P avaiciunas@agpu:/home/avaiciunas/gfs/tts/espnet/egs2/lina/variants/out/*.mp3 mp3s

gt_files=$(patsubst %,mp3s/%_gt.mp3,s01 s02 s03 s04 s05)
make-gt: $(gt_files)
mp3s/s01_gt.mp3: | mp3s
	ffmpeg -i $(corpus)/lab_LIN_SIR_A080_000.wav $@	
mp3s/s02_gt.mp3: | mp3s
	ffmpeg -i $(corpus)/lab_LIN_SIR_A080_001.wav $@	
mp3s/s03_gt.mp3: | mp3s
	ffmpeg -i $(corpus)/lab_LIN_SIR_A080_002.wav $@		
mp3s/s04_gt.mp3: | mp3s
	ffmpeg -i $(corpus)/lab_LIN_SIR_A080_003.wav $@	
mp3s/s05_gt.mp3: | mp3s
	ffmpeg -i $(corpus)/lab_LIN_SIR_A080_004.wav $@	

put-online:
	rsync -P out.html airenas@sinteze.intelektika.lt:~/intelektikaLt-deploy/deploy/c8/tts/html/lina-samples/
	rsync -rP mp3s airenas@sinteze.intelektika.lt:~/intelektikaLt-deploy/deploy/c8/tts/html/lina-samples/


out.html: mp3s display_html.py
	ls -1 mp3s/*_*_*.mp3 | python display_html.py > $@_
	mv $@_ $@

generate_html: out.html

clean:
	rm -f out.html
	rm -rf mp3s

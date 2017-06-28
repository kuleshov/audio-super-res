SINGLE_SPEAKER_DIR=../VCTK-Corpus/wav48/p225

# size of validation set
N_VAL_PHRASES=8

# patch generation
SCA=4
DIM=6000
STR=3000
SR=16000

# ----------------------------------------------------------------------------

patches:
	make vctk-speaker1-interp-train.$(SCA).$(SR).$(DIM).$(STR).h5 vctk-speaker1-interp-val.$(SCA).$(SR).$(DIM).$(STR).h5

clean:
	-rm speaker1-interp-train-files.txt speaker1-interp-val-files.txt
	-rm vctk-speaker1-interp-train.*.$(SR).$(DIM).$(STR).h5 vctk-speaker1-interp-val.*.$(SR).$(DIM).$(STR).h5

# ----------------------------------------------------------------------------
# create dataset for one speaker

vctk-speaker1-interp-train.%.$(SR).$(DIM).$(STR).h5: speaker1-interp-train-files.txt
	python ../prep_vctk.py \
		--interpolate \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale $* \
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR)
	mv $@.tmp $@

vctk-speaker1-interp-val.%.$(SR).$(DIM).$(STR).h5: speaker1-interp-val-files.txt
	python ../prep_vctk.py \
		--interpolate \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale $* \
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR)
	mv $@.tmp $@

speaker1-interp-train-files.txt:
	ls $(SINGLE_SPEAKER_DIR) | head -n -$(N_VAL_PHRASES) > $@

speaker1-interp-val-files.txt:
	ls $(SINGLE_SPEAKER_DIR) | tail -n $(N_VAL_PHRASES) > $@
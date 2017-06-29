SINGLE_SPEAKER_DIR=../VCTK-Corpus/wav48/p225

# patch generation
SCA=4
DIM=8192
STR=4096
SR=16000

# ----------------------------------------------------------------------------

patches:
	make vctk-speaker1-train.$(SCA).$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.$(SCA).$(SR).$(DIM).$(STR).h5

# ----------------------------------------------------------------------------
# create dataset for one speaker

vctk-speaker1-train.%.$(SR).$(DIM).$(STR).h5: speaker1-train-files.txt
	python ../prep_vctk.py \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale $* \
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR) \
		--interpolate \
		--low-pass
	mv $@.tmp $@

vctk-speaker1-val.%.$(SR).$(DIM).$(STR).h5: speaker1-val-files.txt
	python ../prep_vctk.py \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale $* \
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR) \
		--interpolate \
		--low-pass
	mv $@.tmp $@

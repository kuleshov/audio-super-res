SINGLE_SPEAKER_DIR=../VCTK-Corpus/wav48/p225

# size of validation set
N_VAL_PHRASES=8

# patch generation
SCA=2
DIM=6000
STR=3000
SR=16000

# ----------------------------------------------------------------------------

patches:
	make vctk-speaker1-train.$(SCA).$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.$(SCA).$(SR).$(DIM).$(STR).h5

multiscale:
	make vctk-speaker1-train.2.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.2.$(SR).$(DIM).$(STR).h5
	make vctk-speaker1-train.3.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.3.$(SR).$(DIM).$(STR).h5
	make vctk-speaker1-train.4.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.4.$(SR).$(DIM).$(STR).h5
	make vctk-speaker1-train.6.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.6.$(SR).$(DIM).$(STR).h5
	make vctk-speaker1-train.8.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.8.$(SR).$(DIM).$(STR).h5
	make vctk-speaker1-train.10.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.10.$(SR).$(DIM).$(STR).h5
	make vctk-speaker1-train.12.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.12.$(SR).$(DIM).$(STR).h5
	make vctk-speaker1-train.15.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.15.$(SR).$(DIM).$(STR).h5

clean:
	-rm speaker1-train-files.txt speaker1-val-files.txt
	-rm vctk-speaker1-train.*.$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.*.$(SR).$(DIM).$(STR).h5

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
		--stride $(STR)
	mv $@.tmp $@

vctk-speaker1-val.%.$(SR).$(DIM).$(STR).h5: speaker1-val-files.txt
	python ../prep_vctk.py \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale $* \
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR)
	mv $@.tmp $@

speaker1-train-files.txt:
	ls $(SINGLE_SPEAKER_DIR) | head -n -$(N_VAL_PHRASES) > $@

speaker1-val-files.txt:
	ls $(SINGLE_SPEAKER_DIR) | tail -n $(N_VAL_PHRASES) > $@
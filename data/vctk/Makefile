URL:=http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz

# ----------------------------------------------------------------------------
# load corpus

all:
	make VCTK-Corpus

VCTK-Corpus: VCTK-Corpus.tar.gz
	tar -zxvf $<

VCTK-Corpus.tar.gz:
	wget $(URL)
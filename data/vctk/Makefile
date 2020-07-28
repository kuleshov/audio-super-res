URL:=http://www.udialogue.org/download/VCTK-Corpus.tar.gz

# ----------------------------------------------------------------------------
# load corpus

all:
	make VCTK-Corpus

VCTK-Corpus: VCTK-Corpus.tar.gz
	tar -zxvf $<

VCTK-Corpus.tar.gz:
	wget $(URL)

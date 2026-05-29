MAIN = main
PDF = $(MAIN).pdf

.PHONY: all clean arxiv

all: $(PDF)

$(PDF): $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).bbl $(MAIN).blg

arxiv: clean $(PDF)
	mkdir -p arxiv-submission
	cp $(MAIN).tex arxiv-submission/
	tar -czf probes-arxiv.tar.gz -C arxiv-submission .
	@echo "Submission tarball: probes-arxiv.tar.gz"

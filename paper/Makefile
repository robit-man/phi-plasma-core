# Build the arxiv-ready paper PDF.
# Requires: TeX Live (or MacTeX), specifically pdflatex + bibtex.
#
# Usage:
#   make           # build paper.pdf
#   make clean     # remove build artifacts
#   make arxiv     # produce arxiv-submission tarball

MAIN = main
PDF = $(MAIN).pdf

.PHONY: all clean arxiv

all: $(PDF)

$(PDF): $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	@echo ""
	@echo "Built $(PDF). Size: $$(wc -c < $(PDF)) bytes."
	@echo "Pages: $$(pdfinfo $(PDF) 2>/dev/null | grep Pages | awk '{print $$2}' || echo 'install pdfinfo')"

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).synctex.gz
	rm -f *.fdb_latexmk *.fls

arxiv: clean $(PDF)
	mkdir -p arxiv-submission
	cp $(MAIN).tex arxiv-submission/
	cp -r figures arxiv-submission/ 2>/dev/null || true
	tar -czf phi-plasma-core-arxiv.tar.gz -C arxiv-submission .
	@echo "Submission tarball: phi-plasma-core-arxiv.tar.gz"
	@echo "Upload to https://arxiv.org/submit"

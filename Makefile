.PHONY: pdf dev build deploy clean

# Run local dev server
dev:
	yarn start

# Build for production
build:
	yarn build

# Generate PDF from agentic guide via Quarto
# Drops output into static/pdf/ so it's served by the site
pdf:
	quarto render _quarto.yml
	cp _quarto-output/Production-RAG-and-Agentic-Systems.pdf static/pdf/production-rag-agentic-systems.pdf
	@echo "✓ PDF at static/pdf/production-rag-agentic-systems.pdf"

# Deploy to GitHub Pages
# Usage: make deploy msg="Add Chapter 2: Embedding Models"
deploy:
	git add -A
	git commit -m "$(msg)"
	git push origin main
	@echo "✓ Pushed — live at vinayj.com in ~60 seconds"

clean:
	rm -rf build .docusaurus _quarto-output

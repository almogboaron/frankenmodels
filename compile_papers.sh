#!/bin/bash

# Script to compile both versions of the FrankenModels paper

echo "FrankenModels Paper Compilation Script"
echo "====================================="
echo

# Check if IEEE style file exists
if [ -f "IEEEtran.cls" ]; then
    echo "IEEE style file found. Will attempt to compile both versions."
    HAS_IEEE=true
else
    echo "IEEE style file not found. Will only compile the simple version."
    HAS_IEEE=false
fi

# Compile simple version
echo
echo "Compiling simple version (no IEEE style required)..."
pdflatex frankenmodels_article_simple.tex
pdflatex frankenmodels_article_simple.tex  # Run twice for cross-references

if [ $? -eq 0 ]; then
    echo "✅ Simple version compiled successfully: frankenmodels_article_simple.pdf"
else
    echo "❌ Error compiling simple version!"
fi

# Compile IEEE version if style file exists
if [ "$HAS_IEEE" = true ]; then
    echo
    echo "Compiling IEEE conference version..."
    pdflatex frankenmodels_article.tex
    bibtex frankenmodels_article
    pdflatex frankenmodels_article.tex
    pdflatex frankenmodels_article.tex
    
    if [ $? -eq 0 ]; then
        echo "✅ IEEE version compiled successfully: frankenmodels_article.pdf"
    else
        echo "❌ Error compiling IEEE version!"
    fi
fi

echo
echo "Compilation complete!"
echo
echo "If you don't have the IEEE style file and need it, you can download it from:"
echo "https://www.ieee.org/conferences/publishing/templates.html" 
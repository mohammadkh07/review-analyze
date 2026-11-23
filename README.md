# Competitor Review Analyzer

A clean Streamlit web app to analyze customer and competitor reviews for sentiment, context (negation & intensifiers), and top keywords, with filtering and safe CSV export.

## Features
- Paste reviews line-by-line or upload `.txt` / `.csv` files.
- Context-aware keyword scoring with negation and booster handling.
- Summary metrics, sentiment distribution chart, and top positive/negative keywords.
- Sentiment filtering (`All`, `Positive`, `Negative`, `Neutral`) and formula-safe CSV export.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
# Sentiment Analysis on Financial News (Assignment 2)

## Overview

This project implements **sentiment analysis models** on financial news headlines and articles to estimate their potential impact on stock prices. It was built as part of a course assignment, combining **data collection, preprocessing, machine learning, and trading simulation**.

We:

- Collect historical stock price data and S&P 500 index values
- Scrape or process financial news data
- Compute volatility, abnormal returns, and impact scores
- Vectorize news text (bag-of-words, TF-IDF, curated sentiment words)
- Train a multi-layer perceptron (MLP) classifier in PyTorch
- Simulate trading strategies driven by predicted impact scores
- Evaluate results and reflect on improvements

---

## Repository Structure

SentimentAnalysis/

│

├── Assignment_2/

│   ├── phase_1/              # Data collection & preprocessing

│   │   ├── 1_collect_data.py

│   │   ├── 2_calculate_volatility.py

│   │   ├── 3_estimate_impact_scores.py

│   │   ├── 4_identify_and_vectorize.py

│   │   └── testcollect.ipynb

│   │

│   ├── phase_2/              # Modeling

│   │   ├── 1_process.py

│   │   ├── 2_model.py

│   │   ├── 2_training.py

│   │   └── 4_eval.py

│   │

│   ├── phase_3/              # Trading simulation

│   │   ├── 1_trading_rules.py

│   │   └── 2_trading_sim_eval.py

│   │

│   ├── datasets/             # Generated CSV outputs

│   ├── dataset_schema.py

│   ├── test_dataset_schema.py

│   ├── utils.py

│   ├── webscraping.py

│   └── webscraping_headless_parsing.py

│

├── news_datasets/            # Provided input datasets

│   ├── analyst_ratings.csv

│   └── headlines.csv

│

└── README.md

<pre class="overflow-visible!" data-start="1932" data-end="2177"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"></div></pre>

---

## Requirements

- Python 3.9+
- Libraries:
  - `pandas`
  - `numpy`
  - `yfinance`
  - `scikit-learn`
  - `torch`
  - `beautifulsoup4`, `requests` (if scraping full articles)

Install with:

```bash
pip install -r requirements.txt
---
```

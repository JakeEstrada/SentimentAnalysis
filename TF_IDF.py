# tfidf_headlines.py
# Minimal TF-IDF for headlines.csv → saves sparse matrix + vectorizer + feature names

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sktext
import joblib

# ---------- CONFIG ----------
CSV_PATH   = Path("headlines.csv")   # change if stored elsewhere
TEXT_COL   = "headline"              # text column in your CSV
OUT_DIR    = Path("tfidf_out")       # outputs will be saved here

# Vectorizer settings (sane defaults for headlines)
MAX_FEATURES = 80000       # cap vocab size (protect RAM)
NGRAM_RANGE  = (1, 2)      # unigrams + bigrams
MIN_DF       = 3           # drop terms in <3 docs
MAX_DF       = 0.95        # drop terms in >95% of docs
USE_STOPWORDS = True       # set to False to keep every token
DTYPE        = np.float32

# ---------- LOAD ----------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")

df = pd.read_csv(CSV_PATH)
if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not in CSV. Columns: {df.columns.tolist()}")

texts = df[TEXT_COL].fillna("").astype(str).tolist()
print(f"[INFO] Loaded {len(texts)} documents from {CSV_PATH.name}")

# ---------- TF-IDF ----------
stop_words = sktext.ENGLISH_STOP_WORDS if USE_STOPWORDS else None
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF,
    stop_words='english',
    lowercase=True,
    dtype=DTYPE,
    norm="l2",
    sublinear_tf=True,   # log(1+tf)
)

X = vectorizer.fit_transform(texts)  # CSR sparse matrix [num_docs, vocab_size]
print(f"[INFO] TF-IDF shape: {X.shape}  (rows=docs, cols=features)")

# ---------- SAVE ----------
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Sparse matrix
sparse.save_npz(OUT_DIR / "tfidf_matrix.npz", X)

# 2) Vectorizer (to reuse same vocab later)
joblib.dump(vectorizer, OUT_DIR / "tfidf_vectorizer.joblib")

# 3) Feature names (for inspection)
with open(OUT_DIR / "tfidf_feature_names.json", "w") as f:
    json.dump(vectorizer.get_feature_names_out().tolist(), f)

# 4) Tiny preview (nonzeros per doc) to sanity-check sparsity
preview = pd.DataFrame({"row_id": np.arange(X.shape[0]),
                        "nnz": np.diff(X.indptr)})
preview.to_csv(OUT_DIR / "preview_row_sparsity.csv", index=False)

print(f"[OK] Saved artifacts to: {OUT_DIR.resolve()}")

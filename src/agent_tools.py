# --- Make sure these imports are at the top of agent_tools.py ---
import duckdb
import pandas as pd
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import numpy as np
import gdown  # <-- Import gdown
import streamlit as st # <-- Import streamlit
import time
import subprocess
import sys
import logging

# -----------------------------------------------------------------

class DuckDBRunner:
    def __init__(self, parquet_dir="./data/parquet"):
        self.conn = duckdb.connect(database=':memory:')
        self.parquet_dir = parquet_dir
        self._register_tables()

    def _register_tables(self):
        p = Path(self.parquet_dir)
        for f in p.glob("*.parquet"):
            tbl = f.stem
            self.conn.execute(f"CREATE VIEW {tbl} AS SELECT * FROM read_parquet('{str(f)}')")
        print("Registered tables:", [f.stem for f in p.glob("*.parquet")])

    def run(self, sql):
        forbidden = ['INSERT','UPDATE','DELETE','ALTER','DROP']
        if any(tok in sql.upper() for tok in forbidden):
            raise ValueError("Unsafe SQL detected.")
        return self.conn.execute(sql).df()


class FaissRetriever:
    """Safe FAISS retriever with absolute paths and full debug logging."""

    def __init__(self, index_dir="./data/faiss_index", model_name="all-MiniLM-L6-v2"):
        import gdown
        index_dir = Path(index_dir).resolve()
        index_dir.mkdir(parents=True, exist_ok=True)

        DRIVE_INDEX_ID = "1pOx2dcv7i7xR3BSs9r8GLj-UrXd13f3z"
        DRIVE_META_ID  = "1-MqxsGV6-nC22lWcnywArM61DEJGW_l5"

        INDEX_PATH = index_dir / "faiss.index"
        META_PATH  = index_dir / "docs_meta.pkl"
        LOG_FILE = index_dir / "faiss_debug.log"

        def log(msg):
            with open(LOG_FILE, "a") as f:
                f.write(msg + "\n")
            st.write(msg)

        log(f"🔍 CWD: {os.getcwd()}")
        log(f"📁 Using FAISS_DIR: {index_dir}")
        log(f"📄 Expected files:\n - {INDEX_PATH}\n - {META_PATH}")

        # Download missing files
        if not INDEX_PATH.exists() or not META_PATH.exists():
            log("⚠️ FAISS files missing — starting download from Google Drive...")

            try:
                if not INDEX_PATH.exists():
                    log("⬇️ Downloading faiss.index ...")
                    gdown.download(f"https://drive.google.com/uc?id={DRIVE_INDEX_ID}", str(INDEX_PATH), quiet=False)
                if not META_PATH.exists():
                    log("⬇️ Downloading docs_meta.pkl ...")
                    gdown.download(f"https://drive.google.com/uc?id={DRIVE_META_ID}", str(META_PATH), quiet=False)
            except Exception as e:
                log(f"❌ gdown download failed: {e}")
                st.error(f"gdown failed: {e}")
                self.index = None
                self.docs = []
                return

        # Check existence again
        log(f"✅ File presence check: INDEX={INDEX_PATH.exists()} META={META_PATH.exists()}")
        if not INDEX_PATH.exists():
            log(f"❌ Still missing index at {INDEX_PATH}")
            raise FileNotFoundError(f"FAISS index missing after download attempt: {INDEX_PATH}")
        if not META_PATH.exists():
            log(f"❌ Still missing metadata at {META_PATH}")
            raise FileNotFoundError(f"FAISS metadata missing after download attempt: {META_PATH}")

        # Load model and FAISS
        try:
            log(f"🧠 Loading model {model_name}")
            self.model = SentenceTransformer(model_name)
            log("✅ Model loaded successfully.")
        except Exception as e:
            log(f"❌ SentenceTransformer load failed: {e}")
            raise

        try:
            log(f"📦 Loading FAISS index from {INDEX_PATH}")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                self.docs = pickle.load(f)
            log(f"✅ Loaded FAISS index with {len(self.docs)} documents.")
        except Exception as e:
            log(f"❌ FAISS read_index failed: {e}")
            raise

    def retrieve(self, query, top_k=5):
        if self.index is None:
            return [{"text": "FAISS index not available or failed to load.", "meta": {}}]
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        results = []
        for i in I[0]:
            if i < len(self.docs):
                text, meta = self.docs[i]
                results.append({"text": text, "meta": meta})
        return results


def df_to_plot_png(df, title="Chart"):
    # ... (this function remains the same) ...
    plt.figure(figsize=(8,4))
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 2:
        plt.plot(df[numeric_cols[0]], df[numeric_cols[1]])
    elif len(numeric_cols) == 1:
        plt.plot(df[numeric_cols[0]])
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

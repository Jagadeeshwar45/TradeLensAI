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
    """FAISS retriever with detailed debug logging for Streamlit Cloud."""

    def __init__(self, index_dir="./data/faiss_index", model_name="all-MiniLM-L6-v2"):
        index_dir = Path(index_dir).resolve()
        index_dir.mkdir(parents=True, exist_ok=True)

        DRIVE_INDEX_ID = "1pOx2dcv7i7xR3BSs9r8GLj-UrXd13f3z"
        DRIVE_META_ID  = "1-MqxsGV6-nC22lWcnywArM61DEJGW_l5"

        INDEX_PATH = index_dir / "faiss.index"
        META_PATH  = index_dir / "docs_meta.pkl"

        st.write("📂 FAISS index path:", str(INDEX_PATH))
        st.write("📂 FAISS meta path:", str(META_PATH))
        st.write("📁 Current working directory:", os.getcwd())

        # Step 1: Check if gdown is available
        try:
            import gdown
            st.info("✅ gdown successfully imported.")
        except Exception as e:
            st.error(f"❌ gdown import failed: {e}")
            raise

        # Step 2: Download if missing
        if not INDEX_PATH.exists() or not META_PATH.exists():
            st.warning("⚠️ FAISS files missing — starting download from Google Drive...")

            def try_download(url, out_path, label):
                st.write(f"⬇️ Downloading {label} from {url}")
                try:
                    gdown.download(url, str(out_path), quiet=False)
                except Exception as e:
                    st.error(f"❌ gdown.download({label}) failed: {e}")
                finally:
                    if out_path.exists():
                        st.write(f"✅ {label} file size: {out_path.stat().st_size / (1024*1024):.2f} MB")
                    else:
                        st.error(f"❌ {label} not found at {out_path} after download attempt!")

            if not INDEX_PATH.exists():
                try_download(f"https://drive.google.com/uc?id={DRIVE_INDEX_ID}", INDEX_PATH, "faiss.index")
            if not META_PATH.exists():
                try_download(f"https://drive.google.com/uc?id={DRIVE_META_ID}", META_PATH, "docs_meta.pkl")

        # Step 3: Confirm both exist
        if not INDEX_PATH.exists() or not META_PATH.exists():
            st.error("❌ One or both FAISS files are still missing after download!")
            st.write("Index exists:", INDEX_PATH.exists())
            st.write("Meta exists:", META_PATH.exists())
            st.stop()

        # Step 4: Load model and index
        try:
            st.info(f"🧠 Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            st.success("✅ Model loaded.")
        except Exception as e:
            st.error(f"❌ Failed to load SentenceTransformer: {e}")
            raise

        # Step 5: Load FAISS index
        try:
            st.info(f"📦 Loading FAISS index from {INDEX_PATH}...")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                self.docs = pickle.load(f)
            st.success(f"✅ FAISS index loaded successfully with {len(self.docs)} docs.")
        except Exception as e:
            st.error(f"❌ FAISS index load failed: {e}")
            self.index = None
            self.docs = []
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

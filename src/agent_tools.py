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
import gdown
import streamlit as st
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


# -----------------------------------------------------------------
# ✅ Clean version — hides logs from Streamlit UI but keeps backend logging
# -----------------------------------------------------------------
class FaissRetriever:
    """FAISS retriever with hidden Streamlit logs (clean UI)."""

    DRIVE_INDEX_ID = "1pOx2dcv7i7xR3BSs9r8GLj-UrXd13f3z"
    DRIVE_META_ID  = "1-MqxsGV6-nC22lWcnywArM61DEJGW_l5"

    MIN_INDEX_BYTES = 100_000
    MIN_META_BYTES  = 1_000

    def __init__(self, index_dir="./data/faiss_index", model_name="all-MiniLM-L6-v2"):
        self.debug_lines = []
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path  = self.index_dir / "docs_meta.pkl"
        self.model_name = model_name
        self.model = None
        self.index = None
        self.docs = []

        # initialize file logger (silent in UI)
        log_path = self.index_dir / "download_debug.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode="a", encoding="utf-8"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("=== Starting FaissRetriever initialization ===")
        logging.info(f"Index dir: {self.index_dir}")

        try:
            import gdown
            self.gdown = gdown
            logging.info("gdown import OK")
        except Exception as e:
            self.gdown = None
            logging.error(f"gdown not available: {e}")
            self._set_failed("gdown not installed; cannot download FAISS files at runtime.")
            return

        # Check files
        need_download = (not self.index_path.exists()) or (not self.meta_path.exists())
        logging.info(f"Files exist? index:{self.index_path.exists()} meta:{self.meta_path.exists()}")

        if need_download:
            logging.info("FAISS files missing. Attempting gdown download.")
            if self.gdown is None:
                self._set_failed("gdown missing; cannot download FAISS files.")
                return
            try:
                idx_url = f"https://drive.google.com/uc?id={self.DRIVE_INDEX_ID}"
                meta_url = f"https://drive.google.com/uc?id={self.DRIVE_META_ID}"

                if not self.index_path.exists():
                    logging.info(f"Downloading faiss.index from {idx_url}")
                    self.gdown.download(idx_url, str(self.index_path), quiet=False)

                if not self.meta_path.exists():
                    logging.info(f"Downloading docs_meta.pkl from {meta_url}")
                    self.gdown.download(meta_url, str(self.meta_path), quiet=False)

                if self.index_path.stat().st_size < self.MIN_INDEX_BYTES:
                    raise RuntimeError(f"faiss.index too small: {self.index_path.stat().st_size}")
                if self.meta_path.stat().st_size < self.MIN_META_BYTES:
                    raise RuntimeError(f"docs_meta.pkl too small: {self.meta_path.stat().st_size}")

                logging.info("Downloads completed successfully.")
            except Exception as e:
                logging.error(f"Download failed: {e}")
                self._set_failed(f"gdown download failed: {e}")
                return

        # Load embedding model
        try:
            logging.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logging.info("SentenceTransformer loaded successfully.")
        except Exception as e:
            logging.error(f"Model load failed: {e}")
            self._set_failed(f"Failed to load embedding model: {e}")
            return

        # Load FAISS index + docs metadata
        try:
            logging.info(f"Reading FAISS index from {self.index_path} (size {self.index_path.stat().st_size})")
            self.index = faiss.read_index(str(self.index_path))
            logging.info("faiss.read_index succeeded.")
            with open(self.meta_path, "rb") as f:
                self.docs = pickle.load(f)
            logging.info(f"Loaded docs metadata: {len(self.docs)} entries.")
        except Exception as e:
            logging.error(f"Failed to load FAISS index or metadata: {e}")
            self._set_failed(f"Failed to load FAISS index or metadata: {e}")
            return

        logging.info("✅ FAISS index initialized successfully.")

    def _set_failed(self, msg):
        logging.error(msg)
        self.index = None
        self.docs = []

    def retrieve(self, query, top_k=5):
        if self.index is None:
            return [{"text": "FAISS index not available or failed to load. See download_debug.log", "meta": {}}]
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        results = []
        for i in I[0]:
            if i < len(self.docs):
                text, meta = self.docs[i]
                results.append({"text": text, "meta": meta})
        return results


# -----------------------------------------------------------------
def df_to_plot_png(df, title="Chart"):
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

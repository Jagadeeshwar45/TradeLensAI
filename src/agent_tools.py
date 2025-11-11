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
    """
    Minimal: download two files from Google Drive using gdown, then load FAISS index.
    Expects DRIVE_INDEX_ID and DRIVE_META_ID to be accessible via "Anyone with link".
    Writes a debug log to data/faiss_index/download_debug.log for inspection.
    """
    DRIVE_INDEX_ID = "1pOx2dcv7i7xR3BSs9r8GLj-UrXd13f3z"
    DRIVE_META_ID  = "1-MqxsGV6-nC22lWcnywArM61DEJGW_l5"

    # tune these thresholds for sanity checks
    MIN_INDEX_BYTES = 100_000        # 100 KB minimum (raise this if your index is bigger — which it is)
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

        # helper
        def dbg(msg):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            self.debug_lines.append(line)
            try:
                st.write(line)
            except Exception:
                pass
            logging.info(line)

        dbg(f"Starting FaissRetriever init. index_dir={self.index_dir}")

        # 1) ensure gdown available
        try:
            import gdown
            self.gdown = gdown
            dbg("gdown import OK")
        except Exception as e:
            self.gdown = None
            dbg(f"gdown not available: {e}")
            self._flush_debug()
            # continue — we'll show a clear error below if download required

        # 2) if files missing, try to download using gdown
        need_download = (not self.index_path.exists()) or (not self.meta_path.exists())
        dbg(f"Files exist? index:{self.index_path.exists()} meta:{self.meta_path.exists()}")

        if need_download:
            dbg("FAISS files missing. Attempting gdown download.")
            if self.gdown is None:
                dbg("Cannot download: gdown not installed in environment.")
                self._flush_debug()
                self._set_failed("gdown not installed; cannot download FAISS files at runtime.")
                return

            try:
                idx_url = f"https://drive.google.com/uc?id={self.DRIVE_INDEX_ID}"
                meta_url = f"https://drive.google.com/uc?id={self.DRIVE_META_ID}"

                if not self.index_path.exists():
                    dbg(f"Downloading faiss.index from {idx_url} -> {self.index_path}")
                    # gdown.download returns path str on success, or None/raises
                    out = self.gdown.download(idx_url, str(self.index_path), quiet=False)
                    dbg(f"gdown.download returned: {out}")

                if not self.meta_path.exists():
                    dbg(f"Downloading docs_meta.pkl from {meta_url} -> {self.meta_path}")
                    out2 = self.gdown.download(meta_url, str(self.meta_path), quiet=False)
                    dbg(f"gdown.download returned: {out2}")

                # sanity check sizes
                if not self.index_path.exists():
                    dbg("Download finished but faiss.index file still missing.")
                    self._flush_debug()
                    self._set_failed("faiss.index file missing after download attempt.")
                    return
                if self.index_path.stat().st_size < self.MIN_INDEX_BYTES:
                    dbg(f"Downloaded faiss.index too small: {self.index_path.stat().st_size} bytes")
                    self._flush_debug()
                    self._set_failed("faiss.index download appears truncated or too small.")
                    return

                if not self.meta_path.exists():
                    dbg("docs_meta.pkl missing after download.")
                    self._flush_debug()
                    self._set_failed("docs_meta.pkl missing after download attempt.")
                    return
                if self.meta_path.stat().st_size < self.MIN_META_BYTES:
                    dbg(f"Downloaded docs_meta.pkl too small: {self.meta_path.stat().st_size} bytes")
                    self._flush_debug()
                    self._set_failed("docs_meta.pkl appears truncated or too small.")
                    return

                dbg("Downloads completed and passed size checks.")
            except Exception as e:
                dbg(f"Exception during gdown download: {e}")
                self._flush_debug()
                self._set_failed(f"gdown download failed: {e}")
                return

        # 3) load embedding model
        try:
            dbg(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            dbg("SentenceTransformer loaded.")
        except Exception as e:
            dbg(f"Failed to load SentenceTransformer model: {e}")
            self._flush_debug()
            self._set_failed(f"Failed to load embedding model: {e}")
            return

        # 4) load faiss index + docs metadata
        try:
            dbg(f"Reading FAISS index from {self.index_path} (size {self.index_path.stat().st_size})")
            self.index = faiss.read_index(str(self.index_path))
            dbg("faiss.read_index succeeded.")
            dbg(f"Loading docs metadata from {self.meta_path}")
            with open(self.meta_path, "rb") as f:
                self.docs = pickle.load(f)
            dbg(f"Loaded docs metadata: {len(self.docs)} entries (type={type(self.docs)})")
            st.success(f"✅ Loaded FAISS index ({len(self.docs)} docs).")
            self._flush_debug()
        except Exception as e:
            dbg(f"Failed to load FAISS index or metadata: {e}")
            self._flush_debug()
            self._set_failed(f"Failed to load FAISS index or metadata: {e}")
            return

    def _set_failed(self, msg):
        logging.error(msg)
        try:
            st.error(msg)
        except Exception:
            pass
        self.index = None
        self.docs = []

    def _flush_debug(self):
        # write debug lines to a log file so you can inspect after deployment
        try:
            log_path = self.index_dir / "download_debug.log"
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(self.debug_lines) + "\n")
            logging.info(f"Wrote debug log to {log_path}")
        except Exception as e:
            logging.warning(f"Could not write debug log: {e}")

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

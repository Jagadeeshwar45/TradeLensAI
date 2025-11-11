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

# add at top of file
import time
import subprocess
import sys
import logging

# inside file, replace your FaissRetriever class with this
class FaissRetriever:
    def __init__(self, index_dir="./data/faiss_index", model_name="all-MiniLM-L6-v2"):
        index_dir = Path(index_dir).resolve()
        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "docs_meta.pkl"

        # change these to your Drive file IDs (gdown) OR GitHub raw links if not LFS
        DRIVE_INDEX_ID = "1pOx2dcv7i7xR3BSs9r8GLj-UrXd13f3z"
        DRIVE_META_ID  = "1-MqxsGV6-nC22lWcnywArM61DEJGW_l5"
        GDRIVE_INDEX_URL = f"https://drive.google.com/uc?id={DRIVE_INDEX_ID}"
        GDRIVE_META_URL  = f"https://drive.google.com/uc?id={DRIVE_META_ID}"

        # minimal expected sizes (bytes) to detect failed tiny downloads
        MIN_INDEX_BYTES = 10_000_00   # ~100 KB minimal; adjust higher if you know size
        MIN_META_BYTES  = 1_000       # meta small but nonzero

        logging.info(f"FaissRetriever: index_dir={index_dir}")
        os.makedirs(index_dir, exist_ok=True)

        # Try to import gdown; better to have it in requirements.txt
        try:
            import gdown
            has_gdown = True
            logging.info("FaissRetriever: gdown available")
        except Exception:
            gdown = None
            has_gdown = False
            logging.warning("FaissRetriever: gdown not available in environment")

        # Download if missing
        if not index_path.exists() or not meta_path.exists():
            logging.info("FaissRetriever: Missing FAISS files — attempting download/build.")
            # 1) Try Google Drive via gdown
            if has_gdown:
                def try_download(url, out_path, retries=2):
                    for attempt in range(1, retries+1):
                        logging.info(f"Downloading {url} -> {out_path} (attempt {attempt})")
                        try:
                            gdown.download(url, str(out_path), quiet=False)
                        except Exception as e:
                            logging.warning(f"gdown exception: {e}")
                        if out_path.exists() and out_path.stat().st_size > 0:
                            logging.info(f"Downloaded {out_path} size={out_path.stat().st_size}")
                            return True
                        time.sleep(1)
                    return False

                ok_index = True
                ok_meta  = True
                if not index_path.exists():
                    ok_index = try_download(GDRIVE_INDEX_URL, index_path, retries=3)
                if not meta_path.exists():
                    ok_meta = try_download(GDRIVE_META_URL, meta_path, retries=3)

                # sanity size checks
                if ok_index and index_path.exists() and index_path.stat().st_size < MIN_INDEX_BYTES:
                    logging.warning("FaissRetriever: index file too small — download likely failed.")
                    ok_index = False
                if ok_meta and meta_path.exists() and meta_path.stat().st_size < MIN_META_BYTES:
                    logging.warning("FaissRetriever: meta file too small — download likely failed.")
                    ok_meta = False

                if not (ok_index and ok_meta):
                    logging.warning("FaissRetriever: gdown downloads failed or produced tiny files.")

            else:
                logging.warning("FaissRetriever: skipping gdown - not installed.")

            # 2) Try GitHub raw (only works if not LFS pointers)
            if not index_path.exists() or not meta_path.exists():
                try:
                    import requests
                    base_raw = "https://raw.githubusercontent.com/Jagadeeshwar45/TradeLensAI/main/data/faiss_index"
                    for fname, p, min_bytes in [("faiss.index", index_path, MIN_INDEX_BYTES),
                                               ("docs_meta.pkl", meta_path, MIN_META_BYTES)]:
                        if not p.exists():
                            url = f"{base_raw}/{fname}"
                            logging.info(f"Attempting raw GitHub download: {url}")
                            r = requests.get(url, timeout=30)
                            if r.status_code == 200 and len(r.content) > 0:
                                p.write_bytes(r.content)
                                logging.info(f"Wrote {p} ({p.stat().st_size} bytes)")
                            else:
                                logging.warning(f"Raw GitHub fetch failed {url} status={r.status_code}")
                except Exception as e:
                    logging.warning(f"GitHub raw download attempt failed: {e}")

            # 3) Fallback: try to build index by running your script (only if repo contains data and compute available)
            if (not index_path.exists() or index_path.stat().st_size < MIN_INDEX_BYTES) \
               or (not meta_path.exists() or meta_path.stat().st_size < MIN_META_BYTES):
                build_script = Path(__file__).resolve().parent.parent / "src" / "build_vectorstore.py"
                if build_script.exists():
                    logging.info("FaissRetriever: Falling back to building FAISS index using build_vectorstore.py")
                    try:
                        subprocess.run([sys.executable, str(build_script)], check=True, cwd=str(build_script.parent))
                        logging.info("FaissRetriever: build script finished.")
                    except Exception as e:
                        logging.error(f"FaissRetriever: build script failed: {e}")
                else:
                    logging.warning("FaissRetriever: build script not found; cannot build index on host environment.")

        # Now try to load model + index metadata
        self.model = SentenceTransformer(model_name)

        try:
            if not index_path.exists() or not meta_path.exists():
                raise FileNotFoundError(f"Required FAISS files missing: {index_path}, {meta_path}")
            logging.info(f"FaissRetriever: Loading index from {index_path} (size {index_path.stat().st_size})")
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self.docs = pickle.load(f)
            if not isinstance(self.docs, list) or len(self.docs) == 0:
                logging.warning("FaissRetriever: metadata loaded but empty or not a list.")
        except Exception as e:
            logging.error(f"FaissRetriever: Failed to load FAISS index: {e}")
            self.index = None
            self.docs = []

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

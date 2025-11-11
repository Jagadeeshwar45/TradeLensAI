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

#
# --- THIS IS THE MODIFIED CLASS ---
#
class FaissRetriever:
    def __init__(self, index_dir="./data/faiss_index", model_name="all-MiniLM-L6-v2"):
        
        # !!! —- PASTE YOUR DIRECT-DOWNLOAD LINKS HERE --- !!!
        FAISS_INDEX_URL = "https://drive.google.com/uc?id=1pOx2dcv7i7xR3BSs9r8GLj-UrXd13f3z"
        DOCS_META_URL = "https://drive.google.com/uc?id=1-MqxsGV6-nC22lWcnywArM61DEJGW_l5"

        index_dir = Path(index_dir).resolve()
        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "docs_meta.pkl"

        # --- Download-on-boot logic using gdown ---
        if not index_path.exists() or not meta_path.exists():
            print(f"⚠️ FAISS files missing. Attempting to download from Google Drive...")
            os.makedirs(index_dir, exist_ok=True) 
            
            with st.spinner(f"Downloading FAISS index (this may take a moment)..."):
                try:
                    # Download faiss.index
                    if not index_path.exists():
                        print(f"Downloading faiss.index...")
                        gdown.download(url=FAISS_INDEX_URL, output=str(index_path), quiet=False)
                        print("✅ Downloaded faiss.index.")

                    # Download docs_meta.pkl
                    if not meta_path.exists():
                        print(f"Downloading docs_meta.pkl...")
                        gdown.download(url=DOCS_META_URL, output=str(meta_path), quiet=False)
                        print("✅ Downloaded docs_meta.pkl.")
                        
                except Exception as e:
                    print(f"❌ Failed to download files from Google Drive: {e}")
                    st.error(f"Failed to download required index files: {e}")
                    self.index = None
                    self.docs = []
                    return
        # --- End of new logic ---

        self.model = SentenceTransformer(model_name)

        try:
            print(f"✅ Loading FAISS index from {index_path}")
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self.docs = pickle.load(f)
            if len(self.docs) == 0:
                print("⚠️ Loaded FAISS index but metadata is empty.")
        except Exception as e:
            print(f"❌ Failed to load FAISS index from {index_path}: {e}")
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

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
    def __init__(self, index_dir="./data/faiss_index", model_name="all-MiniLM-L6-v2"):
        index_path = Path(index_dir) / "faiss.index"
        meta_path = Path(index_dir) / "docs_meta.pkl"
        self.model = SentenceTransformer(model_name)

        # ✅ Gracefully handle missing FAISS files
        if not index_path.exists() or not meta_path.exists():
            print("⚠️ FAISS index not found. It will need to be built by the main agent.")
            self.index = None
            self.docs = []
            return

        try:
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self.docs = pickle.load(f)
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            self.index = None
            self.docs = []

    def retrieve(self, query, top_k=5):
        if self.index is None:
            return [{"text": "FAISS index not available. Please rebuild it.", "meta": {}}]
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        return [{"text": self.docs[i][0], "meta": self.docs[i][1]} for i in I[0]]

        return [{"text": self.docs[i][0], "meta": self.docs[i][1]} for i in I[0]]

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

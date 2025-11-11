"""
build_vectorstore.py
Embeds (Portuguese) review texts and product descriptions and builds a FAISS index
that can later be queried in English thanks to multilingual embeddings.
"""

import os
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

PARQUET_DIR = "./data/parquet"
OUT_DIR = "./data/faiss_index"
# multilingual model: supports >50 languages, incl. Portuguese + English
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def build_index():
    os.makedirs(OUT_DIR, exist_ok=True)
    reviews_path = Path(PARQUET_DIR) / "reviews.parquet"
    items_path = Path(PARQUET_DIR) / "order_items_denorm.parquet"

    if not reviews_path.exists():
        raise FileNotFoundError("reviews.parquet missing – run etl.py first.")

    reviews = pd.read_parquet(reviews_path)
    items = pd.read_parquet(items_path)

    docs = []
    # collect review texts
    for _, row in reviews.iterrows():
        text = str(row.get("review_comment_message", "")).strip()
        if text:
            docs.append((text, {
                "type": "review",
                "order_id": row.get("order_id"),
                "product_id": row.get("product_id"),
                "review_score": row.get("review_score")
            }))
    # add simple product descriptions
    for _, r in items[["product_id", "product_category_name"]].drop_duplicates().iterrows():
        txt = f"Produto {r['product_id']} categoria {r['product_category_name']}"
        docs.append((txt, {"type": "product", "product_id": r["product_id"]}))

    print(f"Total docs to embed: {len(docs)}")

    model = SentenceTransformer(MODEL_NAME)
    texts = [t for t, _ in docs]
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))
    with open(os.path.join(OUT_DIR, "docs_meta.pkl"), "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ Vectorstore saved to {OUT_DIR}")

if __name__ == "__main__":
    build_index()

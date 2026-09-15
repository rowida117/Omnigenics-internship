"""
Member 3 — Build FAISS Index
Takes the embeddings Member 2 extracted (already saved as .npy files)
and builds a searchable FAISS index from them. Run this any time the
embeddings are regenerated (e.g. after Member 2 retrains the model).

Usage:
    python build_index.py
"""

import os
import numpy as np
import pandas as pd
import faiss


def l2_normalize(embeddings):
    """Normalize each embedding to unit length: ||x|| = 1"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return embeddings / norms


def build_and_save_index(embeddings_path, metadata_path, output_index_path):
    """
    Loads embeddings + metadata, L2-normalizes, builds a FAISS
    IndexFlatIP (inner product = cosine similarity on normalized
    vectors), and saves it to disk.
    """
    print(f"Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(embeddings_path)

    print(f"Loading metadata from {metadata_path} ...")
    metadata = pd.read_csv(metadata_path)

    assert len(embeddings) == len(metadata), (
        f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata rows. "
        "These must be the same length and in the same order."
    )

    print("Normalizing embeddings...")
    embeddings_norm = l2_normalize(embeddings).astype(np.float32)

    dim = embeddings_norm.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)

    assert index.ntotal == len(embeddings_norm), (
        "FAISS count does not match embeddings count!"
    )

    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    faiss.write_index(index, output_index_path)

    print(f"Saved index to {output_index_path}")
    print(f"  Vectors in index: {index.ntotal}")

    return index


if __name__ == "__main__":
    build_and_save_index(
        embeddings_path="embeddings/train_embeddings.npy",
        metadata_path="embeddings/train_metadata.csv",
        output_index_path="embeddings/faiss_index.bin",
    )

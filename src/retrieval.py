"""
Image Retrieval
Turns a raw skin lesion image into a ranked list of visually similar
images from the FAISS index. This is the function Member 4's app calls.
"""

import numpy as np
import torch
from PIL import Image


def l2_normalize(embeddings):
    """Normalize each embedding to unit length: ||x|| = 1"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return embeddings / norms


def extract_embedding(image_path, model, transform, device):
    """
    Extract a 1280-D embedding from a single image.
    Uses eval_transform (no augmentation) for deterministic results.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    model.eval()
    with torch.no_grad():
        embedding = model.get_embedding(tensor)  # [1, 1280]

    return embedding.cpu().numpy().squeeze()  # (1280,)


def retrieve_similar_images(
    image_path, model, transform, index, metadata_df, device="cpu", k=5
):
    """
    Given a path to a skin lesion image, returns the top-k most similar
    images from the retrieval database (the FAISS index), along with
    their metadata and similarity scores.

    Args:
        image_path:   path to the query image (str)
        model:        trained SkinLesionModel (must have get_embedding())
        transform:    eval_transform (no augmentation — deterministic)
        index:        FAISS index built from the training embeddings
        metadata_df:  metadata dataframe aligned with the FAISS index
                      (row i in metadata_df must match vector i in index)
        device:       torch device ('cpu' or 'cuda')
        k:            number of neighbors to retrieve

    Returns:
        pandas DataFrame: top-k rows from metadata_df + 'similarity' column,
        sorted by similarity (highest first).
    """
    embedding = extract_embedding(image_path, model, transform, device)
    embedding = l2_normalize(embedding.reshape(1, -1)).astype(np.float32)

    similarities, indices = index.search(embedding, k)

    results = metadata_df.iloc[indices[0]].copy().reset_index(drop=True)
    results["similarity"] = similarities[0]

    return results

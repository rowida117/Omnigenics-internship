"""
Member 4 — Skin Lesion Image Retrieval App
Streamlit UI: Upload Image -> Preview -> Search -> Retrieve Top-K -> Display

CURRENT STATUS: running on FAKE / placeholder retrieval results.
Members 1 & 2 haven't delivered model.py / best_model.pth yet, so the
`run_retrieval()` function below returns hardcoded fake data instead of
calling the real pipeline. Once their files arrive, only that one
function needs to change — everything else (layout, upload, display,
error handling) is already done and tested.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
MAX_FILE_SIZE_MB = 10
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
TOP_K = 5
ISIC_API = "https://api.isic-archive.com/api/v2/images"

# Set to True once Members 1 & 2 deliver model.py + best_model.pth
# and you swap run_retrieval() to call the real pipeline.
USE_REAL_PIPELINE = True

# Each entry maps a model's display name -> the paths it needs.
# Member 3 builds one FAISS index per embedding set (see build_index.py,
# run once per embedding type). Add/remove rows here as models become
# available; the UI adapts automatically.
MODEL_REGISTRY = {
    "Fine-Tuned (Ours)": {
        "index_path": "embeddings/faiss_index.bin",
        "metadata_path": "embeddings/train_metadata.csv",
        "available": True,
    }
}


@st.cache_data(show_spinner=False)
def fetch_isic_image(isic_id):
    """Fetch one result image from the ISIC API and cache it for this session."""
    isic_id = str(isic_id).strip()
    metadata_response = requests.get(f"{ISIC_API}/{isic_id}/", timeout=20)
    metadata_response.raise_for_status()

    image_url = metadata_response.json()["files"]["full"]["url"]
    image_response = requests.get(image_url, timeout=30)
    image_response.raise_for_status()

    return Image.open(BytesIO(image_response.content)).convert("RGB")


# ══════════════════════════════════════════════════════
# RETRIEVAL — fake version now, real version later
# ══════════════════════════════════════════════════════
def run_retrieval_fake(uploaded_image, model_name, k=TOP_K):
    """
    Placeholder retrieval used while the real model/indices aren't
    available yet. Returns a DataFrame shaped exactly like what
    retrieve_similar_images() will eventually return, so swapping
    this out later requires no changes to the display code below.

    Similarity scores are nudged slightly per model just so the demo
    visibly "looks different" across models — purely cosmetic, delete
    once real results are wired in.
    """
    base_sims = [0.94, 0.92, 0.89, 0.87, 0.85][:k]
    nudge = {
        "Pretrained (ImageNet)": -0.10,
        "Fine-Tuned (Ours)": 0.0,
        "Metric Learning": 0.03,
    }.get(model_name, 0.0)

    fake_results = pd.DataFrame(
        {
            "image_id": [
                "ISIC_0027419",
                "ISIC_0025030",
                "ISIC_0029176",
                "ISIC_0025661",
                "ISIC_0031633",
            ][:k],
            "category": ["malignant", "malignant", "benign", "malignant", "benign"][:k],
            "dx": ["mel", "mel", "nv", "bcc", "nv"][:k],
            "similarity": [round(min(s + nudge, 0.99), 2) for s in base_sims],
        }
    )
    return fake_results


@st.cache_resource
def load_model_resources(model_name):
    """
    Loads (model, transform, index, metadata) for one entry in
    MODEL_REGISTRY. Cached per model_name so switching back and forth
    in the UI doesn't reload from disk every time.

    Wire this up once Member 2's model.py / best_model.pth and
    Member 3's per-model faiss_index_*.bin files exist:
    """
    import torch, faiss
    from torchvision import transforms
    from src.model import SkinLesionModel

    config = MODEL_REGISTRY[model_name]
    checkpoint = torch.load("models/best_model.pth", map_location="cpu")
    model = SkinLesionModel(
        backbone_name=checkpoint["backbone"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    index = faiss.read_index(config["index_path"])
    metadata = pd.read_csv(config["metadata_path"])
    return model, eval_transform, index, metadata

    raise NotImplementedError(f"Real resources for '{model_name}' aren't wired up yet.")


def run_retrieval_real(uploaded_image, model_name, k=TOP_K):
    """
    Real pipeline for a specific model. Same retrieval logic every
    time (retrieve_similar_images) — only which model/index gets
    loaded changes, via load_model_resources(model_name).
    """
    from src.retrieval import retrieve_similar_images  # PIL-image variant

    model, eval_transform, index, metadata = load_model_resources(model_name)
    return retrieve_similar_images(
        uploaded_image,
        model,
        eval_transform,
        index,
        metadata,
        device="cpu",
        k=k,
    )


def run_retrieval(uploaded_image, model_name, k=TOP_K):
    if USE_REAL_PIPELINE:
        return run_retrieval_real(uploaded_image, model_name, k=k)
    return run_retrieval_fake(uploaded_image, model_name, k=k)


def available_models():
    """Models the UI should offer. In demo mode, all are 'available'."""
    if not USE_REAL_PIPELINE:
        return list(MODEL_REGISTRY.keys())
    return [name for name, cfg in MODEL_REGISTRY.items() if cfg["available"]]


# ══════════════════════════════════════════════════════
# VALIDATION / ERROR HANDLING
# ══════════════════════════════════════════════════════
def validate_upload(uploaded_file):
    """Returns (is_valid, error_message)."""
    if uploaded_file is None:
        return False, "No file uploaded."

    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext not in ALLOWED_TYPES:
        return (
            False,
            f"Unsupported file type '.{file_ext}'. Please upload a JPG or PNG image.",
        )

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            False,
            f"Image is too large ({size_mb:.1f} MB). Please upload an image under {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        image.verify()
    except Exception:
        return (
            False,
            "This file couldn't be read as a valid image. Please try a different file.",
        )

    return True, None


# ══════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Skin Lesion Image Retrieval", layout="wide")

st.title("🔬 Skin Lesion Image Retrieval")
st.caption(
    "Upload a skin lesion image to find visually similar cases from our database. "
    "This tool is for research/educational purposes only and is not a medical diagnosis."
)

if not USE_REAL_PIPELINE:
    st.info(
        "⚠️ Running in **demo mode** with placeholder results — the trained model "
        "and search index haven't been connected yet.",
        icon="ℹ️",
    )

st.divider()

col_upload, col_query = st.columns([1, 1])

with col_upload:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image",
        type=ALLOWED_TYPES,
        help=f"Accepted formats: {', '.join(ALLOWED_TYPES)}. Max size: {MAX_FILE_SIZE_MB} MB.",
    )

    models = available_models()
    compare_mode = st.checkbox(
        "Compare all models side by side",
        help="Runs the search against every available model and shows results together, "
        "instead of picking one.",
    )
    selected_model = None
    if not compare_mode:
        selected_model = st.selectbox(
            "Model",
            options=models,
            index=models.index("Fine-Tuned (Ours)")
            if "Fine-Tuned (Ours)" in models
            else 0,
            help="Which embedding model to search with. See the project report for how "
            "these compare on Precision@k / Recall@k / mAP.",
        )

    search_clicked = st.button(
        "🔍 Search", type="primary", disabled=uploaded_file is None
    )

with col_query:
    st.subheader("2. Preview")
    if uploaded_file is not None:
        is_valid, error_message = validate_upload(uploaded_file)
        if is_valid:
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Query image", width=280)
        else:
            st.error(error_message)
    else:
        st.write("No image uploaded yet.")

st.divider()

# ══════════════════════════════════════════════════════
# SEARCH + RESULTS
# ══════════════════════════════════════════════════════
if search_clicked and uploaded_file is not None:
    is_valid, error_message = validate_upload(uploaded_file)

    if not is_valid:
        st.error(f"Cannot search: {error_message}")
    else:
        query_image = Image.open(uploaded_file)
        models_to_run = available_models() if compare_mode else [selected_model]

        results_by_model = {}
        with st.spinner("Searching for similar cases..."):
            for model_name in models_to_run:
                try:
                    results_by_model[model_name] = run_retrieval(
                        query_image, model_name, k=TOP_K
                    )
                except NotImplementedError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong searching with '{model_name}': {e}")

        def render_results_block(results, heading):
            if results is None:
                return
            if len(results) == 0:
                st.warning(f"No similar cases were found ({heading}).")
                return

            st.markdown(f"**{heading}**")
            cols = st.columns(len(results))
            for col, (_, row) in zip(cols, results.iterrows()):
                with col:
                    image_id = row["image_id"]
                    try:
                        result_image = fetch_isic_image(image_id)
                        st.image(
                            result_image,
                            caption=str(image_id),
                            use_container_width=True,
                        )
                    except (requests.RequestException, KeyError, ValueError) as error:
                        st.warning(f"Could not load {image_id}: {error}")

                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:8px; padding:12px; text-align:center;">
                            <div style="font-size:12px; color:#888;">{image_id}</div>
                            <div style="font-weight:600; margin-top:4px;">{row["category"].title()}</div>
                            <div style="font-size:13px; color:#555;">similarity: {row["similarity"]:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            counts = results["category"].value_counts()
            total = len(results)
            summary_line = " · ".join(
                f"{cat.title()}: {cnt}/{total}" for cat, cnt in counts.items()
            )
            st.caption(f"Vote summary — {summary_line}")

        if results_by_model:
            st.subheader("3. Most Similar Cases")

            if compare_mode:
                for model_name in models_to_run:
                    render_results_block(
                        results_by_model.get(model_name), heading=model_name
                    )
                    st.divider()
                st.caption(
                    "Comparing models side by side shows how embedding quality affects retrieval — "
                    "see the project report for full Precision@k / Recall@k / mAP numbers."
                )
            else:
                render_results_block(
                    results_by_model.get(selected_model), heading=selected_model
                )
                st.caption(
                    "This summary reflects the diagnoses of visually similar cases in our "
                    "database, not a diagnosis of the uploaded image."
                )

elif search_clicked and uploaded_file is None:
    st.error("Please upload an image before searching.")

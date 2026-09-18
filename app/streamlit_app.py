"""
Member 4 & Chatbot Integration — Skin Lesion Image Retrieval & AI Assistant App
Streamlit UI: Upload Image -> Search -> Retrieve Top-K -> Display -> AI Chatbot Interpretation
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

from src.chatbot import DermChatbot

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
MAX_FILE_SIZE_MB = 10
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
TOP_K = 5
ISIC_API = "https://api.isic-archive.com/api/v2/images"

USE_REAL_PIPELINE = True

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
# RETRIEVAL — fake version fallback, real version when assets exist
# ══════════════════════════════════════════════════════
def run_retrieval_fake(uploaded_image, model_name, k=TOP_K):
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


def run_retrieval_real(uploaded_image, model_name, k=TOP_K):
    from src.retrieval import retrieve_similar_images

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
    if not USE_REAL_PIPELINE:
        return list(MODEL_REGISTRY.keys())
    return [name for name, cfg in MODEL_REGISTRY.items() if cfg["available"]]


def validate_upload(uploaded_file):
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
# UI SETUP & SESSION STATE
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Skin Lesion Retrieval & AI Assistant", layout="wide", page_icon="🔬")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_results" not in st.session_state:
    st.session_state.latest_results = None

# ══════════════════════════════════════════════════════
# SIDEBAR CONTROL PANEL
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.header("🤖 AI Assistant Settings")
    provider = st.selectbox(
        "Chatbot LLM Provider",
        ["Built-in Grounded Assistant", "OpenAI (GPT-4o-mini)", "Google Gemini"],
        help="Select which AI engine powers the conversational assistant. The Built-in engine works offline without an API key.",
    )

    api_key = None
    if provider != "Built-in Grounded Assistant":
        api_key = st.text_input(
            f"Enter {provider.split()[0]} API Key",
            type="password",
            help="Your API key is used strictly for this session and never stored.",
        )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("🔬 **Skin Cancer Retrieval & AI Assistant System**")
    st.caption("Built for research & educational assessment of dermoscopic lesions.")

# ══════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════
st.title("🔬 Skin Lesion Retrieval & AI Assistant")
st.caption(
    "Upload a skin lesion image to find visually similar cases from our database, "
    "and interact with our grounded AI Assistant to interpret results and learn skin health concepts."
)

if not USE_REAL_PIPELINE:
    st.info(
        "⚠️ Running in **demo mode** with placeholder retrieval results.",
        icon="ℹ️",
    )

st.divider()

# ══════════════════════════════════════════════════════
# SECTION 1 & 2: UPLOAD AND PREVIEW
# ══════════════════════════════════════════════════════
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
        help="Runs search against every available embedding model.",
    )
    selected_model = None
    if not compare_mode:
        selected_model = st.selectbox(
            "Model Architecture",
            options=models,
            index=models.index("Fine-Tuned (Ours)")
            if "Fine-Tuned (Ours)" in models
            else 0,
        )

    search_clicked = st.button(
        "🔍 Search Similar Cases", type="primary", disabled=uploaded_file is None
    )

with col_query:
    st.subheader("2. Query Preview")
    if uploaded_file is not None:
        is_valid, error_message = validate_upload(uploaded_file)
        if is_valid:
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Query image", width=260)
        else:
            st.error(error_message)
    else:
        st.write("No image uploaded yet.")

st.divider()

# ══════════════════════════════════════════════════════
# SECTION 3: SEARCH EXECUTION & RESULTS
# ══════════════════════════════════════════════════════
if search_clicked and uploaded_file is not None:
    is_valid, error_message = validate_upload(uploaded_file)

    if not is_valid:
        st.error(f"Cannot search: {error_message}")
    else:
        query_image = Image.open(uploaded_file)
        models_to_run = available_models() if compare_mode else [selected_model]

        results_by_model = {}
        with st.spinner("Searching vector index for similar cases..."):
            for model_name in models_to_run:
                try:
                    results_by_model[model_name] = run_retrieval(
                        query_image, model_name, k=TOP_K
                    )
                except Exception as e:
                    st.error(f"Something went wrong searching with '{model_name}': {e}")

        if results_by_model:
            # Store first result dataframe for the chatbot context
            first_key = list(results_by_model.keys())[0]
            st.session_state.latest_results = results_by_model[first_key]

# Render Search Results
if st.session_state.latest_results is not None:
    st.subheader("3. Most Visually Similar Database Cases")

    def render_results_block(results, heading):
        if results is None or len(results) == 0:
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
                except Exception as error:
                    st.warning(f"Could not load {image_id}")

                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; border-radius:8px; padding:10px; text-align:center;">
                        <div style="font-size:12px; color:#888;">{image_id}</div>
                        <div style="font-weight:600; margin-top:4px;">{row['category'].title()}</div>
                        <div style="font-size:13px; color:#555;">similarity: {row['similarity']:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        counts = results["category"].value_counts()
        total = len(results)
        summary_line = " · ".join(f"{cat.title()}: {cnt}/{total}" for cat, cnt in counts.items())
        st.caption(f"Vote summary — {summary_line}")

    render_results_block(st.session_state.latest_results, heading=selected_model or "Retrieved Cases")
    st.caption("This summary reflects visual feature similarity to database cases, NOT a medical diagnosis.")
    st.divider()

# ══════════════════════════════════════════════════════
# SECTION 4: INTERACTIVE AI CHATBOT & EXPLAINER
# ══════════════════════════════════════════════════════
st.subheader("💬 AI Assistant & Result Interpreter")
st.caption("Ask questions about your retrieval search results, dermatological terms, ABCDE rules, or skin health guidance.")

# Quick Action Suggestion Buttons
st.markdown("**Quick Prompts:**")
qp_col1, qp_col2, qp_col3, qp_col4 = st.columns(4)

selected_prompt = None
if qp_col1.button("📊 Explain search results"):
    selected_prompt = "Explain my search results"
if qp_col2.button("🩺 What is the ABCDE rule?"):
    selected_prompt = "What is the ABCDE rule?"
if qp_col3.button("🔬 Does similarity = cancer %?"):
    selected_prompt = "Does a high similarity score mean high cancer probability?"
if qp_col4.button("⚠️ What should I do next?"):
    selected_prompt = "What are the recommended next steps?"

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
user_input = st.chat_input("Type your question here (e.g. 'Why is border irregularity important?')...")

prompt_to_process = selected_prompt or user_input

if prompt_to_process:
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)

    # Generate Bot Response
    bot_engine = DermChatbot(provider=provider, api_key=api_key)
    with st.chat_message("assistant"):
        with st.spinner("Thinking & analyzing context..."):
            response_text = bot_engine.respond(
                user_query=prompt_to_process,
                chat_history=st.session_state.messages,
                retrieval_df=st.session_state.latest_results,
            )
            st.markdown(response_text)

    # Store assistant response in session
    st.session_state.messages.append({"role": "assistant", "content": response_text})

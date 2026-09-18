# Skin Cancer Image Retrieval System

## Overview
This project implements a **Content-Based Image Retrieval (CBIR)** system for dermoscopic skin lesions using the **ISIC Archive dataset**. Rather than acting as a simple "black box" diagnostic classifier, this system provides a highly interpretable "second opinion" tool for clinicians. When given a query image of a skin lesion, the system instantly retrieves historically verified, visually and pathologically similar cases from a vast database.

## Key Features
* **AI Feature Extraction:** Utilizes state-of-the-art Deep Learning Convolutional Neural Networks (like EfficientNet) to convert 2D skin lesion images into mathematical, high-dimensional embeddings.
* **Rapid Similarity Search:** Employs Facebook AI Similarity Search (FAISS) to instantly search through thousands of embeddings using Cosine Similarity.
* **Clinical Interpretability:** Empowers dermatologists by presenting the closest visual matches alongside their ground-truth diagnoses, making the AI's decision-making process transparent and reliable.
* **Interactive UI:** A Streamlit-based web application allowing users to upload a query image, run the similarity search, and review the top-K retrieved similar cases.

## Dataset
The system is built upon the **International Skin Imaging Collaboration (ISIC)** dataset—the global gold standard for dermoscopic image analysis. Key diagnostic classes include:
- Melanoma
- Nevus
- Basal Cell Carcinoma
- (And other various benign/malignant conditions)

The dataset undergoes extensive preprocessing, including resizing, normalization, and augmentation, with strict train/test split boundaries to prevent patient-level data leakage.

## Project Structure
```text
├── embeddings/        # Contains FAISS index and generated feature embeddings (.npy)
├── model/             # Saved model checkpoints (e.g., efficientnet_b0_best.pth)
├── src/               # Core source code for preprocessing, dataset, model, and retrieval
├── results/           # Evaluation metrics and generated plots
├── app/               # Streamlit application (streamlit_app.py)
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rowida117/Omnigenics-internship.git
   cd Omnigenics-internship
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   Launch the Streamlit user interface locally:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Evaluation Metrics
The retrieval engine's accuracy is heavily evaluated using standard metrics, including:
- **Precision@K** (e.g., P@1, P@5)
- **Recall@K**
- **Mean Average Precision (mAP)**

Performance boosts are analyzed by comparing raw pretrained models against fine-tuned/metric-learning optimized embedding spaces.

## Future Work
- Incorporation of textual patient metadata (e.g., age, anatomical site) into the retrieval weighting algorithm.
- Advanced handling for underrepresented skin tones to reduce potential AI bias.

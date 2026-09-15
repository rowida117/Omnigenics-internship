
# Team Division — Skin Cancer Image Retrieval

### Overall pipeline

```text
                 ISIC Dataset
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
   Data Preparation          Metadata/Labels
          │                       │
          └───────────┬───────────┘
                      ↓
              Feature Extraction
                      ↓
                Embeddings
                      ↓
                FAISS Index
                      ↓
              Image Retrieval
                      ↓
             Evaluation + UI
```

---

## Member 1 — Dataset & Preprocessing

### Responsibility

**Data collection, cleaning, EDA, and preprocessing**

### Tasks

1. Select the exact ISIC dataset/collection.
2. Download the images and metadata.
3. Organize the dataset.
4. Analyze:

   * Number of images
   * Classes
   * Class distribution
   * Missing values
   * Image dimensions
   * Duplicate images
5. Clean the dataset.
6. Create train/validation/test splits.
7. Make sure the split avoids **data leakage**, preferably at patient/lesion level where identifiers permit.
8. Implement the preprocessing pipeline:

   * Resize
   * Normalize
   * Basic augmentation
9. Create PyTorch `Dataset` and `DataLoader`.

### Deliverables

```text
data/
├── raw/
├── processed/
├── metadata.csv
└── splits/

src/
├── dataset.py
└── preprocessing.py
```

### Main output

A clean dataset ready to be consumed by the model.

---

# Member 2 — Deep Learning / Feature Extraction

### Responsibility

**Build the model that converts a skin lesion image into an embedding**

This is the core AI part.

### Tasks

1. Select baseline architecture:

   * EfficientNet-B0
   * ResNet50
   * DenseNet121

2. Load ImageNet pretrained weights.

3. Build the feature extractor:

```text
Image
  ↓
CNN
  ↓
Global Average Pooling
  ↓
Embedding
```

For example:

```text
224 × 224 × 3
       ↓
EfficientNet
       ↓
1280-dimensional embedding
```

4. Train a classification baseline:

```text
Image
 ↓
EfficientNet
 ↓
Embedding
 ↓
Classifier
 ↓
Diagnosis
```

5. Fine-tune the model on ISIC.

6. Save the best model.

7. Implement embedding extraction.

8. Generate embeddings for the entire retrieval database.

### Deliverables

```text
src/
├── model.py
├── train.py
└── extract_embeddings.py

models/
└── best_model.pth

embeddings/
└── embeddings.npy
```

### Main output

A function like:

```python
embedding = extract_embedding(image)
```

which converts an image into a feature vector.

---

# Member 3 — Image Retrieval & Evaluation

### Responsibility

**Build the actual retrieval engine and prove that it works**

This person takes Member 2's embeddings and turns them into a searchable database.

### Tasks

1. Normalize embeddings.

2. Build the FAISS index.

```text
All Image Embeddings
        ↓
      FAISS
        ↓
Vector Index
```

3. Implement query retrieval:

```python
results = search(query_image, k=5)
```

4. Calculate similarity using:

* Cosine similarity
* Inner product / FAISS
* Euclidean distance as an optional comparison

5. Retrieve metadata associated with each image.

6. Implement evaluation.

### Metrics

At minimum:

```text
Precision@1
Precision@5
Recall@5
Recall@10
mAP
```

Also generate:

* Confusion matrix if using retrieval-based classification
* Per-class retrieval performance
* Similarity distributions

### Important experiment

Compare:

```text
Model A: Pretrained EfficientNet
             vs
Model B: Fine-tuned EfficientNet
             vs
Model C: Metric Learning
```

If you have enough time, Member 3 can implement **Triplet Loss or Supervised Contrastive Learning** with Member 2 providing the model architecture.

### Deliverables

```text
src/
├── build_index.py
├── retrieval.py
└── evaluation.py

embeddings/
└── faiss_index.bin

results/
├── metrics.csv
├── plots/
└── retrieval_examples/
```

### Main output

A function:

```python
results = retrieve_similar_images(query_image, k=5)
```

returning something like:

```text
Image ID       Diagnosis       Similarity
ISIC_001234    melanoma        0.94
ISIC_005678    melanoma        0.92
ISIC_009876    nevus           0.89
...
```

---

# Member 4 — Application / Integration

### Responsibility

**Turn everything into a usable application**

This person should **not work independently from the beginning**. They can start the UI while Members 1–3 build the pipeline.

### Tasks

1. Build Streamlit interface.

2. Implement:

```text
Upload Image
     ↓
Preview Image
     ↓
Search
     ↓
Retrieve Top-K
```

3. Display:

* Query image
* Top-5 similar images
* ISIC ID
* Diagnosis
* Similarity score

4. Add optional diagnosis summary.

For example:

```text
Retrieved Cases

Melanoma: 4/5
Nevus:    1/5
BCC:      0/5
```

5. Connect:

```text
UI
 ↓
Preprocessing
 ↓
Model
 ↓
Embedding
 ↓
FAISS
 ↓
Results
```

6. Handle errors:

* Wrong file type
* Very large image
* Missing model
* Invalid image
* No results

7. Improve UI/UX.

8. Prepare deployment.

### Deliverables

```text
app/
└── streamlit_app.py

requirements.txt
README.md
```

### Main output

A working application:

```text
┌──────────────────────────────────────┐
│     Skin Lesion Image Retrieval      │
├──────────────────────────────────────┤
│                                      │
│       Upload Skin Image              │
│                                      │
│          [ Choose File ]             │
│                                      │
│             [SEARCH]                 │
│                                      │
├──────────────────────────────────────┤
│           Query Image                │
│                                      │
├──────────────────────────────────────┤
│       Most Similar Cases             │
│                                      │
│  Image 1    Image 2    Image 3       │
│  Melanoma   Melanoma   Nevus         │
│  0.94       0.92       0.89          │
└──────────────────────────────────────┘
```

---

# How the 4 members connect

This is the important part.

### Member 1 → Member 2

Member 1 provides:

```python
train_loader
val_loader
test_loader
```

and the cleaned dataset.

↓

### Member 2 → Member 3

Member 2 provides:

```text
best_model.pth
embeddings.npy
metadata.csv
```

↓

### Member 3 → Member 4

Member 3 provides:

```text
faiss_index.bin
retrieval.py
```

with a simple interface such as:

```python
retrieve_similar_images(image, k=5)
```

↓

### Member 4

Integrates everything:

```text
Streamlit
    ↓
Member 1 preprocessing
    ↓
Member 2 model
    ↓
Member 2 embedding extraction
    ↓
Member 3 FAISS
    ↓
Member 3 retrieval
    ↓
UI results
```

---

# Suggested timeline

If you have around **6 weeks**, I would organize it like this:

| Week  | Member 1                 | Member 2        | Member 3            | Member 4          |
| ----- | ------------------------ | --------------- | ------------------- | ----------------- |
| **1** | Dataset + EDA            | Research models | Research retrieval  | UI design         |
| **2** | Cleaning + preprocessing | Baseline model  | Retrieval prototype | Basic Streamlit   |
| **3** | Final splits             | Fine-tuning     | FAISS               | Connect model     |
| **4** | Data analysis            | Embeddings      | Evaluation          | Retrieval UI      |
| **5** | Support experiments      | Metric learning | Compare models      | Final integration |
| **6** | Documentation            | Model analysis  | Results/plots       | Deployment + demo |

---

# Workload balance

There is one issue with the original division: **Member 2 and Member 3 have more technically difficult work** than Member 1.

So I would make the responsibilities slightly broader:

### Member 1

**Data + EDA + preprocessing + data augmentation**

### Member 2

**Deep learning + fine-tuning + embedding extraction**

### Member 3

**Retrieval + metric learning + evaluation**

### Member 4

**Application + integration + deployment + documentation**

This is more balanced.

---

# Who should do what in the final presentation?

You can also divide the presentation naturally:

### Member 1 — Problem & Dataset

Explain:

* Skin cancer problem
* Why image retrieval
* ISIC dataset
* Dataset statistics
* Preprocessing

### Member 2 — Deep Learning

Explain:

* CNN architecture
* Transfer learning
* Fine-tuning
* Embeddings
* Why embeddings are useful for retrieval

### Member 3 — Retrieval & Evaluation

Explain:

* Cosine similarity
* FAISS
* Top-K retrieval
* Precision@K
* Recall@K
* mAP
* Experimental comparison

### Member 4 — Application

Explain:

* System architecture
* Streamlit interface
* User workflow
* Demo
* Limitations
* Future work

---

# One important recommendation

Don't make the project **only**:

> "Upload image → find similar images."

That can look relatively simple.

Make the research contribution the comparison:

```text
                    Retrieval
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
  ImageNet         Fine-tuned      Metric
  pretrained       EfficientNet    Learning
        │              │              │
        ↓              ↓              ↓
    Embeddings      Embeddings     Embeddings
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                     FAISS
                       ↓
                  Top-K Retrieval
                       ↓
                 Compare Results
```

Then your project has a clear **research question**, measurable experiments, and an actual application.

**Suggested project title:**

> **Deep Learning-Based Content Image Retrieval for Skin Lesion Analysis Using the ISIC Archive**

Or, if you implement metric learning:

> **Deep Metric Learning for Content-Based Retrieval of Dermoscopic Skin Lesions**

The second title is stronger academically.

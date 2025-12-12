# Handwritten Answer Grading System



```markdown
# Handwritten Answer Grading System

This repository provides a complete, reproducible pipeline for automated evaluation of handwritten student answers using OCR-based text extraction, Sentence-BERT semantic similarity, and a machine-learning-based scoring mechanism.  
The goal is to enable reliable, interpretable grading on constrained hardware using modular components that can be improved or replaced independently.

---

## 1. Problem Statement

Traditional automated assessment tools struggle with handwritten responses due to:

- variability in handwriting styles  
- OCR inaccuracies  
- inconsistent semantic interpretation  
- lack of domain grounding  
- absence of explainable grading components  

General-purpose LLMs can evaluate text, but they cannot directly process handwriting, and their evaluations often lack transparency or determinism. Furthermore, schools and institutions require grading systems that are:

- interpretable  
- reproducible  
- data-efficient  
- modular  
- able to integrate with existing workflows  

**Objective:**  
Develop a deterministic, modular pipeline that can:

1. Extract handwritten text from scanned PDFs  
2. Compare student answers to expected solutions using semantic embeddings  
3. Predict a score through a supervised ML model  
4. Generate structured, interpretable feedback  
5. Produce outputs in a standardized JSON format suitable for downstream systems  

---

## 2. Abstract

This project implements an OCR-to-evaluation pipeline for automated grading of handwritten academic responses. The workflow begins by converting PDF pages to images, applying a handwriting OCR model (EasyOCR), and producing clean text output.  
Semantic similarity is computed using **Sentence-BERT (all-MiniLM-L6-v2)**, which embeds both student answers and expected answers into a dense vector space, enabling cosine-similarity-based comparison.

A **RandomForestRegressor** trained on handcrafted similarity–grade pairs predicts a final score, while a lightweight rule-based layer generates human-interpretable feedback.  
Outputs are returned in JSON format, enabling downstream integration with dashboards, LMS systems, or analytics modules.

This pipeline achieves efficient, transparent evaluation without requiring high-capacity GPUs or proprietary models.

---

## 3. Architecture Overview

### 3.1 Text-Based Architecture Diagram

```

Handwritten PDF
│
▼
PDF-to-Image Converter (pdf2image)
│
▼
OCR Model (EasyOCR)
│
▼
Extracted Text
│
▼
Semantic Encoder (Sentence-BERT)
│
▼
Cosine Similarity Computation
│
▼
RandomForest Grade Predictor
│
▼
Feedback Generator
│
▼
JSON Output

```

This architecture prioritizes modularity—each component can be independently replaced (e.g., transformer-based handwriting recognition, larger embedding models, learned feedback generator).

```
## 3.2 Architecture Diagrams

### Overall System Architecture  
![Architecture](assets/architecture.png)

### End-to-End Workflow  
![Workflow](assets/flowchart.png)

### Module Interaction (Sequence Diagram)  
![Sequence](assets/sequence.png)

---
```
## 4. Dataset

This project does not rely on a large supervised dataset; instead, it uses:

- OCR output from handwritten PDFs  
- A reference expected answer  
- A handcrafted dataset mapping similarity scores to grade labels  

Example training data used for the RandomForest model:

json
[
  {
    "similarity_score": 0.90,
    "grammar_score": 90,
    "final_grade": 9
  }
]
````

Dataset Characteristics:

* Small, interpretable, synthetic dataset
* Suitable for establishing baseline behavior
* Designed to be replaced by larger institutional datasets
* Captures core evaluation dimensions: semantic similarity and linguistic clarity

---

## 5. Methodology

### 5.1 OCR Component

We use **EasyOCR**, which supports handwriting detection and provides text predictions with minimal configuration.

### 5.2 Preprocessing

* Convert PDF pages to images
* Normalize color channels
* Thresholding for cleaner OCR
* Concatenate text across pages

### 5.3 Semantic Encoder: Sentence-BERT

Model: `sentence-transformers/all-MiniLM-L6-v2`
Capabilities:

* Efficient embedding generation
* Cosine similarity interpretation
* Sentence-level semantic understanding

### 5.4 Similarity Computation

Cosine similarity ∈ [0,1] serves as the core evaluation metric.

### 5.5 Grade Prediction Model

Model: **RandomForestRegressor**

Configuration:

* 100 estimators
* Input features: similarity score, grammar score
* Output: continuous grade

### 5.6 Feedback Generator

A deterministic rule-based system:

* High similarity → positive reinforcement
* Medium similarity → constructive advice
* Low similarity → detailed improvement guidance

---

## 6. Repository Structure

```
handwritten-grading-system/
│
├── images/
│   ├── architecture.png
│   ├── flowchart.png
│   └── sequence.png
│
├── handwriting_grader.py
├── requirements.txt
└── README.md
```

---

## 7. Reproducibility

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the System in Colab

```python
!python3 handwriting_grader.py
```

### Output Location

All results are printed and returned as JSON.

---

## 8. Results

### 8.1 Quantitative (Synthetic Baseline)

| Metric                     | Value                             |
| -------------------------- | --------------------------------- |
| OCR Word Accuracy          | Moderate (depends on handwriting) |
| SBERT Similarity Stability | High                              |
| Prediction Variance        | Low                               |
| End-to-End Latency (T4)    | < 2s                              |

---

### 8.2 Qualitative Improvements

After integrating semantic similarity, answers receive evaluations that are:

* more aligned with conceptual correctness
* less sensitive to lexical variance
* more interpretable to end users

---

### 8.3 Failure Cases

* Poor OCR quality due to highly stylized handwriting
* Answers with correct reasoning but unusual phrasing
* Very long responses truncated by model max length
* Grammar score currently static and not model-driven

---

## 9. Limitations

* No large-scale supervised dataset
* Grammar scoring not learned
* OCR accuracy bottlenecks the entire pipeline
* No rubric-aligned scoring
* Current system supports only one expected answer per question

---

## 10. Future Work

* Integrate a handwriting transformer model (TrOCR)
* Learn grammar and structure scoring via neural models
* Expand to multi-question assessments
* Add rubric-aware scoring mechanisms
* Apply RLHF or DPO to align grading to teacher preferences
* Build a production-ready FastAPI inference layer
* Extend dataset for institution-level fine-tuning

---



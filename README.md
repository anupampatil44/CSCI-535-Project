# Multimodal Sarcasm Detection
### CSCI 535 Course Project — USC  
Instructor: Prof. Mohammad Soleymani

Multimodal sarcasm detection using **text, audio, and visual cues** from conversational video data in the MUStARD dataset.

---

## Overview

This project explores multimodal learning approaches for sarcasm detection by combining:

- Text embeddings
- Audio prosodic + deep speech features
- Visual facial landmark representations

We implement and compare multiple fusion strategies including:

- Early Fusion
- Late Fusion
- Attention-Based Fusion
- Framewise Temporal Modeling

---

## Dataset

**MUStARD (Multimodal Sarcasm Detection Dataset)**

- 690 video clips
- Balanced sarcastic / non-sarcastic labels
- TV-show conversational data
- Modalities:
  - Text transcripts + context
  - Audio
  - Video

---

## Feature Extraction

### Text
- BERT-base embeddings
- Granite-Embedding-278M-Multilingual
- 768D representations

### Audio
- WavLM / wav2vec2 embeddings
- MFCCs
- Pitch & RMS energy statistics
- Framewise temporal embeddings

### Visual
- MediaPipe Attention Mesh / BlazeFace
- 478 facial landmarks per frame
- PCA-based temporal aggregation

---

## Fusion Architectures

### Early Fusion
Concatenate multimodal embeddings before classification using MLPs and Bi-GRUs.

### Late Fusion
Independent modality encoders followed by meta-classifier fusion.

### Attention Fusion
Transformer-based cross-modal attention with temporal modeling.

---

## Results

| Method | F1 Score |
|---|---|
| Early Fusion | 0.750 |
| Attention Fusion | 0.730 |
| Attention Fusion (Framewise) | 0.727 |
| Late Fusion | 0.686 |
| Gemini 2.0 Pro | 0.833 |

---

## Tech Stack

- PyTorch
- HuggingFace Transformers
- MediaPipe
- Librosa
- Parselmouth
- Scikit-learn
- OpenCV

---

## Repository Structure

```bash
.
├── audio/
├── visual/
├── text/
├── fusion_models/
├── scripts/
├── notebooks/
└── results/
```

---

## Setup

```bash
git clone https://github.com/anupampatil44/CSCI-535-Project.git
cd CSCI-535-Project

pip install -r requirements.txt
```

---


---
```

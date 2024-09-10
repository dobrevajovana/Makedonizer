# 📚 MACEDONIZER: The Macedonian Transformer Language Model

**Authors:** Jovana Dobreva, Tashko Pavlov, Kostadin Mishev, Monika Simjanoska, Stojancho Tudzarski, Dimitar Trajanov, Ljupcho Kocarev  
**Institution:** Ss. Cyril and Methodius University in Skopje, Faculty of Computer Science and Engineering

## 🚀 Introduction

**MACEDONIZER** is the first Macedonian language model based on the RoBERTa architecture. Pre-trained on a 6.5 GB corpus of Macedonian text, the model demonstrates state-of-the-art performance on several Natural Language Processing (NLP) tasks including **Sentiment Analysis (SA)**, **Named Entity Recognition (NER)**, and **Natural Language Inference (NLI)**.

This repository contains the code for training, evaluating, and using the MACEDONIZER model, including the NLI code for inference tasks.

## ✨ Features

- 🔍 **Pre-trained RoBERTa-based model**: Adapted for the Macedonian language with specialized tokenization and fine-tuning on Macedonian text.
- 🎯 **Supports multiple NLP tasks**: Sentiment Analysis, Named Entity Recognition, and Natural Language Inference.
- 🌐 **Publicly available on Hugging Face**: Use the model for further fine-tuning and inference tasks.

## 🧠 Model Overview

MACEDONIZER is built on top of the **RoBERTa** transformer architecture and fine-tuned on a corpus of Macedonian text from various sources, including news articles and Wikipedia. The model outperforms cross-lingual models such as XLM-RoBERTa on multiple Macedonian-specific NLP tasks.

## 📊 Datasets

### Training Data
The pre-training corpus consists of the following:
- 📰 **News Corpus**: 4.95 GB from various news websites in Macedonia.
- 📖 **Macedonian Corpus**: 1.1 GB of public domain texts from OSCAR.
- 🌍 **Macedonian Wikipedia**: 373.3 MB dump of the Macedonian Wikipedia.

### Evaluation Tasks
The model is evaluated on:
- 💬 **Sentiment Analysis (SA)**: A dataset of Macedonian financial news labeled as positive or negative.
- 🏷 **Named Entity Recognition (NER)**: Recognizing entities such as locations, people, and organizations in Macedonian text.
- 🔄 **Natural Language Inference (NLI)**: Predicting relationships between sentences, such as entailment, contradiction, or neutrality.

## 🤗 Hugging Face Model

MACEDONIZER is publicly available on Hugging Face. You can use the model directly from their hub:
[MACEDONIZER on Hugging Face](https://huggingface.co/macedonizer/mk-roberta-base)

## ⚙️ Getting Started

### Prerequisites

- 🐍 Python 3.8+
- 🔥 PyTorch
- 🤗 Hugging Face `transformers`
- 📊 Scikit-learn

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/MACEDONIZER.git
cd MACEDONIZER
```

## 🏆 Results

MACEDONIZER achieves the following results on the evaluation tasks:

| Task   | Accuracy | Precision | Recall | F1 Score | MCC   |
|--------|----------|-----------|--------|----------|-------|
| SA     | 90.7%    | 88.1%     | 96.3%  | 90.6%    | 0.814 |
| NER    | 94.2%    | 94.0%     | 94.2%  | 94.1%    | 0.818 |
| NLI    | 77.2%    | 77.3%     | 77.2%  | 77.2%    | 0.658 |

### 🔧 Usage

To use MACEDONIZER for Natural Language Inference (NLI) tasks, follow the example below:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "macedonizer/mk-roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input sentences
premise = "Примерна реченица на македонски."
hypothesis = "Ова е реченица која следува."
inputs = tokenizer(premise, hypothesis, return_tensors="pt")

# Inference
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
label = ["entailment", "neutral", "contradiction"][predictions]
print(f"Prediction: {label}")
```


## Acknowledgments
The development of this model was partially funded by the Faculty of Computer Science and Engineering at Ss. Cyril and Methodius University in Skopje.

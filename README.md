# Supervised Deep Learning for Cross-Lingual Question Answering in Low-Resource Language

This repository focuses on **Extractive Question Answering (QA)** for **Urdu**, a low-resource language spoken by over 70 million people. By leveraging state-of-the-art multilingual Transformer models (mBERT and XLM-RoBERTa), we reproduce benchmarks on the **UQA dataset** (an Urdu translation of SQuAD 2.0 via the EATS method) and extend the baseline with specific deep learning experiments, including Parameter-Efficient Fine-Tuning (PEFT) and cross-lingual transfer learning.

---

## 📌 Project Overview
- **Objective:** Establish and improve robust extractive QA systems for Urdu text using deep representation learning.
- **Dataset:** UQA dataset, generated from SQuAD 2.0 using the **EATS** (Enclose to Anchor, Translate, Seek) method with Meta's SeamlessM4T Large model.
- **Core Models:** `bert-base-multilingual-cased` (mBERT) and `xlm-roberta-base` (XLM-R).
- **Key Techniques:** Sliding-window tokenization, Low-Rank Adaptation (LoRA), Zero-Shot Cross-Lingual Transfer, and Sequential Cross-Lingual Transfer.

---

## 🧪 Experiments Carried Out

We designed and executed four distinct experiments to test the boundaries of these models:

1. **mBERT Configuration Correction (Exp 1):** Fixed core configuration flaws in previous setups (e.g., implementing missing sliding-window tokenization for contexts $> 512$ tokens), achieving a **+6.53 EM** and **+6.48 F1** performance boost.
2. **XLM-R Parameter-Efficient Fine-Tuning (Exp 2):** Applied **LoRA** ($r=16, \alpha=32$) to fine-tune XLM-RoBERTa base. While full fine-tuning performed better due to learning rate sensitivity, LoRA shrunk the checkpoint size from **1.1 GB to just 19.2 MB**.
3. **Zero-Shot Transfer to Hindi (Exp 3):** Evaluated the Urdu LoRA-adapted model directly on **XQUAD Hindi** without any Hindi training. It achieved **49.83 EM / 65.10 F1**, demonstrating strong cross-linguistic generalization across structurally similar South Asian scripts.
4. **Sequential Cross-Lingual Transfer (Exp 4):** Evaluated warming up XLM-R on English SQuAD before downstream fine-tuning on Urdu UQA. 

---

## 📊 Core Performance Summary

Evaluation was conducted using standard SQuAD evaluation metrics (**Exact Match (EM)** and token-level **F1-Score**) on the answerable-only validation subset ($11,169$ examples):

| Model / Configuration | Training Setup | Exact Match (EM) | F1-Score |
| :--- | :--- | :---: | :---: |
| **mBERT** (Paper Baseline) | Reported Benchmark | 45.50% | 64.72% |
| **mBERT** (Fixed Config - Exp 1) | Full Fine-Tune | **56.81%** | **70.00%** |
| **XLM-RoBERTa base** (Baseline) | Reported Benchmark | 65.67% | 78.00% |
| **XLM-RoBERTa base** (Reproduced)| Full Fine-Tune | **64.45%** | **77.14%** |
| **XLM-R + LoRA** (Exp 2) | PEFT ($r=16$) | 43.49% | 59.97% |
| **XLM-R (SQuAD $\rightarrow$ UQA)** (Exp 4)| Sequential Transfer | 62.85% | 76.12% |

### 🌍 Zero-Shot Cross-Lingual Performance
* **XLM-R (SQuAD Only $\rightarrow$ Urdu Validation):** `49.06% EM / 63.61% F1` *(Zero Urdu training instances seen!)*
* **XLM-R UQA LoRA $\rightarrow$ XQUAD Hindi:** `49.83% EM / 65.10% F1` 

---

## ⚙️ Environment & Hyperparameters

### Hardware Setup
* Infrastructure: Kaggle Notebook Environments
* Accelerator: Dual GPU ($2 \times \text{NVIDIA T4}$, 15 GiB VRAM each)
* CPU Memory: ~30 GiB RAM

### Primary Hyperparameters
* **Max Sequence Length:** 384 (with Doc Stride: 128)
* **Optimizer:** AdamW (Weight Decay: 0.01)
* **Effective Batch Size:** 32 (mBERT) | 256 (LoRA) | 128 (Sequential Transfer)
* **Precision:** FP16 Mixed Precision
* **Seed:** 42

---

## 🛠️ Repository Pipeline Structure

The project code follows the dataset pipeline schema of the baseline framework:
1. `Clean.ipynb` - Processes raw SQuAD 2.0 JSONs, isolates long contexts, standardizes Unicode symbols, and injects anchor delimiters (`..`).
2. `Translate.ipynb` - Orchestrates the Meta `seamlessM4T_large` text-to-text mode (`eng` $\rightarrow$ `urd`) keeping anchors mapped properly.
3. `Dataset Generator.ipynb` - Discards temporary delimiters and reconstructs a clean character-indexed Hugging Face `DatasetDict`.
4. `uqa-mbert-xlmroberta.ipynb` - The primary training script mapping character span indices to token offsets and running the training loop via Hugging Face `Trainer`.

---

## 👥 Authors
* **M. Abdullah Bin Asim** - ([i232621@isb.nu.edu.pk](mailto:i232621@isb.nu.edu.pk))
* **Maimoona Hannan** - ([i232544@isb.nu.edu.pk](mailto:i232544@isb.nu.edu.pk))
* **Fatima Tu Zahra** - ([i232616@isb.nu.edu.pk](mailto:i232616@isb.nu.edu.pk))

*Department of Data Science & Artificial Intelligence, FAST-NUCES, Islamabad, Pakistan.*

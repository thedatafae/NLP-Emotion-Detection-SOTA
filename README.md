# 🎭 Emotion Classification: From Statistical ML to SOTA Transformers

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **"Love" is not just "Joy" with more intensity.** — *How I used Deep Learning to fix semantic ambiguity in text classification.*

## 📌 Overview
This project benchmarks classical Machine Learning algorithms against Deep Learning architectures for multi-class emotion detection. 

The core challenge of this dataset was the semantic overlap between **Love** and **Joy**. Statistical models (TF-IDF) failed to distinguish them because they share identical vocabularies. By implementing **DistilBERT**, I introduced contextual understanding, breaking the 90% accuracy ceiling and achieving **93.8% SOTA performance**.

## 📊 Key Results
| Model | Feature Engineering | Accuracy | Key Observation |
| :--- | :--- | :--- | :--- |
| **Multinomial NB** | Bag of Words | 63% | Fails on negative context. |
| **Logistic Regression** | TF-IDF (1,2-grams) | 84% | Good baseline, lacks deep context. |
| **Linear SVC** | TF-IDF + Oversampling | **90%** | **Best Classical Model.** Hit a math limit on Love/Joy overlap. |
| **DistilBERT** | Raw Text (Transformers) | **94%** | **SOTA.** Solved the context problem via Attention mechanisms. |

---

## 📉 The "Aha!" Moment: Error Analysis

### 1. The Bottleneck (Linear SVM)
Using TF-IDF, the model looks at isolated words. It sees "happy", "smile", "tender" and gets confused.
* **Result:** High confusion between *Love* and *Joy*.
* **Accuracy Cap:** ~90%

![Place your SVM Confusion Matrix Here](https://github.com/thedatafae/NLP-Emotion-Detection-SOTA/raw/main/Charts/SVM_Confusion_Matrix.png)

### 2. The Solution (DistilBERT)
BERT reads the sentence *bidirectionally*. It understands that "I cherish you" (Love) is structurally different from "I am happy for you" (Joy), even if the words are positive.
* **Result:** drastic reduction in False Positives.
* **Final Accuracy:** ~94%

![Place your BERT Confusion Matrix Here](https://github.com/thedatafae/NLP-Emotion-Detection-SOTA/raw/main/Charts/BERT_Confusion_Matrix.png)

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `simpletransformers` (HuggingFace), `scikit-learn`, `pandas`, `seaborn`, `imblearn`
* **Environment:** Google Colab (T4 GPU)

## 📂 Project Structure
```bash
├── data/
│   └── train.txt                 # The dataset
├── notebooks/
│   └── Benchmarking_Emotion_Classification.ipynb   # Full code (Preprocessing -> SVM -> BERT)
└── README.md
```
---

## 🚀 Usage

1. **Clone the Repo**
   ```bash
   git clone [git clone [https://github.com/thedatafae/Emotion-Classification-Benchmark.git](https://github.com/thedatafae/Emotion-Classification-Benchmark.git))
    ```
2. **Install Dependencies**
   ```bash
   pip install simpletransformers scikit-learn pandas
   ```
3. **Run the notebook in Jupyter or Google Colab.**
---
## 👤 Author
**Faizan Ahmed Khan**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/thedatafae)

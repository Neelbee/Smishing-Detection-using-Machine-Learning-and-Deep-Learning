# Smishing-Detection-using-Machine-Learning-and-Deep-Learning
A deep learning–based system for detecting smishing (SMS phishing) messages using a hybrid CNN + BERT model, achieving 99.86% accuracy and 99.45% F1-score, outperforming traditional ML models.

# 📱 Smishing Detection using Hybrid CNN + BERT  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![DeepLearning](https://img.shields.io/badge/Model-CNN%2BBERT-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.86%25-brightgreen.svg)

> **A deep learning–based solution to detect smishing (SMS phishing) messages using a hybrid CNN + BERT model, achieving 99.86% accuracy and 99.45% F1-score.**

---

## 🚀 Overview  

This project leverages **Machine Learning (ML)** and **Deep Learning (DL)** to detect fraudulent SMS messages.  
A hybrid **CNN + BERT** model captures both contextual and spatial text features, significantly outperforming traditional ML classifiers.

---

## 🧩 Key Highlights  

- 🤖 **Hybrid CNN + BERT** for contextual + convolutional feature extraction  
- 📊 Benchmarked with traditional ML models — `SVM`, `Random Forest`, `LSTM`  
- 🧹 Robust preprocessing: **SMOTE**, **TF-IDF**, **tokenization**, **lemmatization**  
- 🌐 **Flask-based web app** with **Bootstrap UI** for real-time SMS classification  
- 🎯 **Performance:** 99.86% Accuracy | 99.45% F1-Score  

---

## 🧠 Dataset  

Collected and preprocessed **13K+ SMS samples** from:  
- 📂 [Kaggle Smishing Dataset](https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set)  
- 📂 [Mendeley SMS Spam Collection](https://data.mendeley.com/datasets/f45bkkt8pr/1)  

**Preprocessing Steps:**  
- Text cleaning (HTML tags, punctuation removal)  
- Lemmatization & Tokenization  
- TF-IDF vectorization for ML models  
- BERT tokenization for hybrid model  
- SMOTE to balance imbalanced data  

---

## 📈 Model Comparison  

| Model          | Accuracy | F1-Score |
|----------------|-----------|-----------|
| Random Forest  | 97.42%    | 97.10%    |
| SVM            | 96.83%    | 96.20%    |
| LSTM           | 98.76%    | 98.40%    |
| **CNN + BERT** | **99.86%** | **99.45%** |

---

## 🧰 Tech Stack  

**Languages & Frameworks:**  
- Python, Flask, HTML, CSS, JavaScript, Bootstrap  

**Libraries:**  
- TensorFlow / PyTorch  
- Transformers (Hugging Face)  
- scikit-learn, NLTK, imbalanced-learn  
- Pandas, NumPy  

---

## ⚙️ Installation & Usage  

### 1️⃣ Clone the Repository  

bash
git clone https://github.com/yourusername/smishing-detection.git

## 2️⃣ Install Dependencies

pip install -r requirements.txt

## 3️⃣ Run the Flask Web App

python app.py

cd smishing-detection


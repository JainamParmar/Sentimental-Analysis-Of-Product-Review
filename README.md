# 🎯 Sentiment Analysis System

## 📌 Project Overview

This project is an end-to-end **Natural Language Processing (NLP)** system that analyzes customer reviews and classifies them into:

* **Positive**
* **Negative**
* **Neutral**

It combines **Machine Learning (ML)** techniques and a **rule-based NLP model (VADER)**, and is deployed through an interactive **Streamlit web application**.

---

## 🚀 Problem Statement

Analyzing customer reviews manually is:

* Time-consuming
* Expensive
* Inconsistent (subjective bias)

### ✅ Solution

Build an automated system that:

* Processes raw text
* Extracts meaningful features
* Predicts sentiment with confidence
* Works in real-time

---

## 🧠 System Architecture

```
Input Review
   ↓
Text Preprocessing
   ↓
Feature Extraction (TF-IDF)
   ↓
Model Training (ML Models)
   ↓
Prediction
   ↓
Streamlit Web App (UI)
```

---

## 🔍 Key Components

### 1. Text Preprocessing

Raw text is cleaned using:

* Lowercasing
* Removing URLs & emails
* Removing special characters
* Tokenization
* Stopword removal
* Lemmatization

**Example:**

```
Input:  "This product is AMAZING!!!"
Output: "product amazing"
```

---

### 2. Feature Engineering

We convert text into numbers using:

### 👉 TF-IDF (Term Frequency – Inverse Document Frequency)

* Assigns importance to words
* Reduces weight of common words
* Increases weight of rare, meaningful words

---

### 3. Machine Learning Models

We trained and compared:

| Model                        | Description                  |
| ---------------------------- | ---------------------------- |
| Logistic Regression          | Fast and accurate (selected) |
| Naive Bayes                  | Simple baseline model        |
| Support Vector Machine (SVM) | High performance but slower  |

**Selected Model:** Logistic Regression

---

### 4. Rule-Based Model (VADER)

The deployed web app uses:

* **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
* No training required
* Works well for short texts and reviews

---

### 5. Sentiment Scoring

VADER provides a **compound score**:

* Range: **-1 to +1**
* Interpretation:

  * ≥ 0.05 → Positive
  * ≤ -0.05 → Negative
  * Otherwise → Neutral

---

## 🌐 Web Application (Streamlit)

### Features:

* 🔍 Single review analysis
* 📊 Batch CSV analysis
* 📈 Visualization (charts & metrics)
* 📥 Download results

### Run the App:

```
streamlit run app.py
```

---

## ⚙️ Installation

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Download NLTK Data

```
python download_nltk.py
```

### 3. Run Notebook (Training)

```
jupyter notebook sentiment_analysis.ipynb
```

### 4. Run Web App

```
streamlit run app.py
```

---

## 📊 Model Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~88%  |
| Precision | ~87%  |
| Recall    | ~88%  |
| F1 Score  | ~87%  |

---

## 🧪 Example Predictions

| Review                     | Sentiment | Confidence |
| -------------------------- | --------- | ---------- |
| "This product is amazing!" | Positive  | 92%        |
| "Terrible quality"         | Negative  | 88%        |
| "It is okay"               | Neutral   | 75%        |

---

## ⚠️ Limitations

* Small dataset (not production scale)
* Cannot detect sarcasm
* Limited context understanding
* VADER may misclassify complex sentences

---

## 🚀 Future Improvements

* Use large real-world datasets (Amazon, Twitter)
* Implement deep learning models (LSTM, BERT)
* Add multilingual support
* Improve UI with dashboards

---

## 🧰 Tech Stack

* Python
* Scikit-learn
* NLTK
* Pandas & NumPy
* Matplotlib
* Streamlit

---

## 🎯 Conclusion

This project demonstrates a complete NLP pipeline from raw text to real-time sentiment prediction using both machine learning and rule-based approaches.

---

## 👨‍💻 Author

ML Mini Project Team

---

## 📌 Note

This project is designed for academic learning and demonstration purposes. For real-world deployment, larger datasets and advanced models are recommended.

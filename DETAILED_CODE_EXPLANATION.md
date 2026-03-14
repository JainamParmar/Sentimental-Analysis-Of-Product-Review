# 🔍 DETAILED CODE EXPLANATION & WALKTHROUGH

## Table of Contents
1. [Text Preprocessing](#text-preprocessing)
2. [Feature Engineering](#feature-engineering)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Prediction System](#prediction-system)
6. [Streamlit Application](#streamlit-application)

---

## Text Preprocessing

### Why Preprocessing?

Raw text contains many elements that confuse machine learning models:
- Extra spaces and punctuation
- Capitalization inconsistencies
- Special characters and URLs
- Common words that don't add sentiment information
- Multiple forms of the same word (running, runs, ran)

**Solution**: Clean and standardize text before feeding to ML models

### Complete Preprocessing Pipeline

```python
class TextPreprocessor:
    """
    Handles all text preprocessing steps
    """
    
    def __init__(self):
        # Load English stopwords (common words)
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw text input
        
        Returns:
            str: Cleaned and processed text
        """
        
        # STEP 1: LOWERCASE
        # Convert all characters to lowercase
        # "This IS Amazing!" → "this is amazing!"
        # Why? Treat 'Amazing' and 'amazing' as the same word
        text = text.lower()
        
        # STEP 2: REMOVE URLS
        # Find and remove URLs like www.example.com, http://...
        # "Check www.example.com for more" → "Check for more"
        # Why? URLs don't contribute to sentiment
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # STEP 3: REMOVE EMAILS
        # Find and remove email addresses
        # "Contact me@example.com" → "Contact"
        # Why? Email addresses are metadata, not sentiment
        text = re.sub(r'\S+@\S+', '', text)
        
        # STEP 4: REMOVE SPECIAL CHARACTERS & NUMBERS
        # Keep only alphabetic characters and spaces
        # "Love!!! Amazing... #BestEver" → "Love Amazing BestEver"
        # Why? Special characters add noise without semantic meaning
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # STEP 5: TOKENIZATION
        # Split text into individual words (tokens)
        # "love amazing fantastic" → ["love", "amazing", "fantastic"]
        # Why? Process words individually for lemmatization
        tokens = word_tokenize(text)
        
        # STEP 6: REMOVE STOPWORDS
        # Filter out common words that don't carry sentiment
        # Common stopwords: the, is, and, or, a, an, in, on, at, to, from, etc.
        # ["the", "product", "is", "amazing"] → ["product", "amazing"]
        # Why? Stopwords appear in all sentiments and add noise
        tokens = [
            word for word in tokens 
            if word not in self.stop_words and len(word) > 1
        ]
        
        # STEP 7: LEMMATIZATION
        # Reduce words to their base/root form
        # "running" → "run", "runs" → "run", "ran" → "run"
        # "better" → "good", "goods" → "good"
        # Why? Treat different forms of the same word as the same feature
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into a single string
        return ' '.join(tokens)
```

### Example Walk-Through

```python
# Original messy review
original = "This product is ABSOLUTELY AMAZING!!! Check www.example.com #BestEver @user"

# Step by step transformation:
# 1. Lowercase
#    "this product is absolutely amazing!!! check www.example.com #besterever @user"

# 2. Remove URLs
#    "this product is absolutely amazing!!! check #besterever @user"

# 3. Remove special characters
#    "this product is absolutely amazing check besterever user"

# 4. Tokenize
#    ["this", "product", "is", "absolutely", "amazing", "check", "besterever", "user"]

# 5. Remove stopwords
#    ["product", "absolutely", "amazing", "check", "besterever", "user"]

# 6. Lemmatize
#    ["product", "absolutely", "amazing", "check", "best", "ever", "user"]

# Final cleaned text
cleaned = "product absolutely amazing check best ever user"
```

### Why Each Step Matters

| Step | Effect | Impact |
|------|--------|--------|
| Lowercase | "AMAZING" = "amazing" | Reduces vocabulary size |
| Remove URLs | No irrelevant data | Focuses on content |
| Tokenize | Split into words | Enables individual processing |
| Remove Stopwords | Reduces noise | Improves model efficiency |
| Lemmatize | "running" = "run" | Better feature representation |

---

## Feature Engineering

### Problem: Text to Numbers

ML models require numerical input, not text. We need to convert text to numerical vectors.

### Solution: TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=100,           # Keep only top 100 terms
    min_df=1,                   # Minimum documents (no filter here)
    max_df=0.9,                 # Maximum 90% of documents
    ngram_range=(1, 2),         # Use single words (1-grams) and pairs (2-grams)
    stop_words='english'        # Remove English stopwords
)

# FIT: Learn vocabulary from training data
# This creates the feature dictionary
tfidf_vectorizer.fit(training_reviews)

# TRANSFORM: Convert reviews to feature vectors
X_train = tfidf_vectorizer.transform(training_reviews)
X_test = tfidf_vectorizer.transform(test_reviews)
```

### Understanding TF-IDF

**TF-IDF = Term Frequency × Inverse Document Frequency**

```
TF (Term Frequency) = How often a word appears in a document
IDF (Inverse Document Frequency) = How rare the word is across all documents

Example:
- Word "love" appears in 80% of documents → Low IDF (common)
- Word "defective" appears in 5% of documents → High IDF (rare)

Result: 
- "love" gets lower weight
- "defective" gets higher weight
- Better representation of sentiment
```

### Mathematical Formula

```
TF-IDF(word, document) = TF(word, document) × IDF(word)

where:
TF(word, document) = count(word) / count(all words in document)
IDF(word) = log(total documents / documents containing word)
```

### Example: Three Reviews

```
Reviews:
1. "love amazing product"
2. "hate terrible defective"
3. "love good quality"

Vocabulary: {love, amazing, product, hate, terrible, defective, good, quality}

TF-IDF Matrix:
       love  amazing  product  hate  terrible  defective  good  quality
Rev1:  0.45   0.63     0.57   0.0    0.0       0.0       0.0    0.0
Rev2:  0.0    0.0      0.0   0.58   0.58      0.73      0.0    0.0
Rev3:  0.45   0.0      0.0   0.0    0.0       0.0       0.68   0.68
```

### Why TF-IDF > CountVectorizer

**CountVectorizer (Simple Word Counts):**
```python
# Count how many times each word appears
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

Result:
"love amazing" → [1, 1, 0, 0]  (love appears 1 time, amazing appears 1 time)
"love amazing amazing" → [1, 2, 0, 0]  (love 1 time, amazing 2 times)
```

**Problem**: Treats all words equally. Common words dominate.

**TF-IDF (Weighted Word Importance):**
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

Result:
"love amazing" → [0.45, 0.63, 0, 0]  (amazing more important than love)
"love amazing amazing" → [0.32, 0.74, 0, 0]  (weights normalized)
```

**Advantage**: Considers word importance. Better for classification.

---

## Model Training

### Three Models Compared

#### 1. Logistic Regression ⭐ **SELECTED**

```python
from sklearn.linear_model import LogisticRegression

# Initialize model
model = LogisticRegression(
    max_iter=200,              # Max iterations to converge
    multi_class='multinomial',  # For 3+ classes
    random_state=42            # Reproducibility
)

# Train on training data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Get probability scores
probabilities = model.predict_proba(X_test)
```

**How It Works:**
```
1. Find optimal weights β₀, β₁, β₂, ... βₙ

2. For prediction:
   z = β₀ + β₁×feature₁ + β₂×feature₂ + ... + βₙ×featureₙ

3. Apply sigmoid function (squashes to 0-1):
   P(positive) = 1 / (1 + e^(-z))

4. Output: Probability for each class
```

**Why Logistic Regression?**
- ✅ Fast training (milliseconds)
- ✅ Fast prediction (real-time)
- ✅ High accuracy for text classification
- ✅ Provides probability scores
- ✅ Easy to interpret
- ✅ Works well with TF-IDF features
- ✅ Production-ready

---

#### 2. Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

# Initialize and train
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**How It Works:**
```
P(Sentiment | Features) = P(Features | Sentiment) × P(Sentiment) / P(Features)

Uses Bayes' theorem: likelihood × prior / evidence

Assumes: Each feature is independent
(This assumption is usually not true, but works well in practice)
```

**Pros:**
- Very fast training
- Good with sparse data
- Works well with high-dimensional features

**Cons:**
- Independence assumption not realistic
- Lower accuracy than Logistic Regression

---

#### 3. Support Vector Machine (SVM)

```python
from sklearn.svm import LinearSVC

# Initialize and train
model = LinearSVC(
    max_iter=1000,     # Iterations
    random_state=42,
    dual=False         # Optimization method
)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

**How It Works:**
```
Finds optimal hyperplane (decision boundary) that:
1. Maximizes margin (distance) between classes
2. Minimizes misclassifications

More complex decision boundary than Logistic Regression
```

**Pros:**
- Powerful classifier
- Good for high-dimensional data
- Can handle non-linear problems

**Cons:**
- Slower training
- No probability scores directly
- Less interpretable

---

## Model Evaluation

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Negative', 'Neutral'])

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Negative', 'Neutral'],
            yticklabels=['Positive', 'Negative', 'Neutral'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

**Example Result:**
```
                 Predicted
             Positive  Negative  Neutral
Actual
Positive        4         0        0     ← Correctly predicted 4 positives
Negative        0         3        0     ← Correctly predicted 3 negatives
Neutral         0         1        1     ← Predicted 1 correctly, 1 wrong
```

### Key Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ACCURACY: Overall correctness
# (TP + TN) / (TP + TN + FP + FN)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # 0.8667 = 86.67%

# PRECISION: When we predict positive, how often correct?
# TP / (TP + FP)
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")

# RECALL: Of actual positives, how many did we find?
# TP / (TP + FN)
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.4f}")

# F1-SCORE: Harmonic mean of precision and recall
# 2 × (Precision × Recall) / (Precision + Recall)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1:.4f}")
```

### Detailed Classification Report

```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)

# Output:
#           precision    recall  f1-score   support
# Negative       0.83      0.83      0.83         3
# Neutral        0.50      0.50      0.50         2
# Positive       1.00      1.00      1.00         4
# accuracy                          0.83         9
# macro avg      0.78      0.78      0.78         9
# weighted avg   0.83      0.83      0.83         9
```

---

## Prediction System

### Complete Prediction Pipeline

```python
class SentimentAnalyzer:
    """End-to-end sentiment analysis"""
    
    def __init__(self, model, vectorizer, preprocessor):
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
    
    def predict(self, review_text):
        """
        Predict sentiment for a single review
        
        Args:
            review_text (str): Customer review
        
        Returns:
            dict: Sentiment and confidence
        """
        
        # STEP 1: PREPROCESS
        # Clean and normalize the input text
        cleaned_review = self.preprocessor.preprocess(review_text)
        # Input:  "This product is AMAZING!!!"
        # Output: "product amazing"
        
        # STEP 2: VECTORIZE
        # Convert text to numerical features using pre-trained vectorizer
        feature_vector = self.vectorizer.transform([cleaned_review])
        # Output: Array of shape (1, 100) with TF-IDF values
        
        # STEP 3: PREDICT
        # Get predicted class (Positive, Negative, or Neutral)
        sentiment = self.model.predict(feature_vector)[0]
        # Output: "Positive"
        
        # STEP 4: GET PROBABILITIES
        # Calculate confidence scores for each class
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(feature_vector)[0]
            confidence = float(np.max(probabilities))
            class_probabilities = dict(zip(self.model.classes_, probabilities))
        else:
            confidence = None
            class_probabilities = None
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
```

### Usage Example

```python
# Initialize analyzer
analyzer = SentimentAnalyzer(
    model=trained_model,
    vectorizer=tfidf_vectorizer,
    preprocessor=text_preprocessor
)

# Predict single review
result = analyzer.predict("This product is amazing!")
print(result)

# Output:
# {
#     'sentiment': 'Positive',
#     'confidence': 0.92,
#     'class_probabilities': {
#         'Positive': 0.92,
#         'Negative': 0.05,
#         'Neutral': 0.03
#     }
# }

# Interpret results
if result['confidence'] > 0.8:
    print("✓ High confidence prediction")
elif result['confidence'] > 0.6:
    print("⚠ Medium confidence prediction")
else:
    print("❌ Low confidence prediction")
```

---

## Streamlit Application

### Application Structure

```python
# app.py - Streamlit web interface

import streamlit as st
import pickle
import pandas as pd

# 1. CONFIGURATION
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="wide"
)

# 2. LOAD MODELS (cached for performance)
@st.cache_resource
def load_models():
    """Load pre-trained models once"""
    with open('model/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, vectorizer, preprocessor

model, vectorizer, preprocessor = load_models()

# 3. HEADER
st.markdown("<h1>😊 Sentiment Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("---")

# 4. TABS
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "Analytics"])

# 5. TAB 1: SINGLE REVIEW ANALYSIS
with tab1:
    st.subheader("Analyze a Single Review")
    
    # Input text area
    review = st.text_area(
        "Enter a product review:",
        placeholder="Example: This product is amazing!",
        height=100
    )
    
    # Analyze button
    if st.button("Analyze Review"):
        # Preprocess
        cleaned = preprocessor.preprocess(review)
        
        # Vectorize
        features = vectorizer.transform([cleaned])
        
        # Predict
        sentiment = model.predict(features)[0]
        
        # Get confidence
        proba = model.predict_proba(features)[0]
        confidence = np.max(proba)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Confidence", f"{confidence:.0%}")
        
        # Show probabilities
        prob_dict = dict(zip(model.classes_, proba))
        st.bar_chart(prob_dict)
```

### Key Streamlit Features

```python
# INPUT WIDGETS
text = st.text_area("Label", placeholder="...")
number = st.number_input("Label", min_value=0, max_value=100)
button = st.button("Click Me")

# DISPLAY
st.write("Simple text")
st.markdown("# Markdown text")
st.metric("Label", value, delta)
st.bar_chart(data)
st.dataframe(df)

# LAYOUT
col1, col2 = st.columns(2)
with col1:
    st.write("Left column")
with col2:
    st.write("Right column")

# CACHING (for performance)
@st.cache_resource
def expensive_computation():
    return result
```

---

## Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    USER INPUT (Streamlit)                     │
│           "This product is absolutely amazing!"               │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                   TEXT PREPROCESSING                          │
│  1. Lowercase: "this product is absolutely amazing"           │
│  2. Remove special: "this product is absolutely amazing"      │
│  3. Tokenize: ["this", "product", "is", "absolutely"]        │
│  4. Remove stopwords: ["product", "absolutely", "amazing"]   │
│  5. Lemmatize: ["product", "absolute", "amazing"]            │
│  Result: "product absolute amazing"                          │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION (TF-IDF)                 │
│  Input: "product absolute amazing"                           │
│  Vocabulary: [love, amazing, product, ...]                   │
│  Output: [0.45, 0.63, 0.28, 0, 0, ...]  (1×100 vector)      │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│              MODEL PREDICTION (Logistic Regression)           │
│  Input: [0.45, 0.63, 0.28, ...]                             │
│  Compute: z = β₀ + β₁×0.45 + β₂×0.63 + ...                 │
│  Apply sigmoid: P(positive) = 1 / (1 + e^(-z))              │
│  Get probabilities: {Positive: 0.92, Negative: 0.05, ...}   │
│  Select max: POSITIVE                                        │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                    DISPLAY RESULT                             │
│            😊 POSITIVE (92% confident)                        │
│  Positive: 92% | Negative: 5% | Neutral: 3%                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Summary of Key Code Sections

| Component | File | Key Function | Time |
|-----------|------|--------------|------|
| Preprocessing | preprocessing.py | `TextPreprocessor.preprocess()` | <1ms |
| Vectorization | model_training.py | `TfidfVectorizer.transform()` | <1ms |
| Prediction | predict.py | `model.predict()` | <1ms |
| Web Interface | app.py | `streamlit run app.py` | - |

---

**🎓 Now you understand the complete pipeline!**


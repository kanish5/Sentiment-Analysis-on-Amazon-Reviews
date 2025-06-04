# 🔍 Sentiment Analysis on Amazon Product Reviews

This project uses Natural Language Processing (NLP) to analyze customer reviews and classify them as **Positive**, **Neutral**, or **Negative**. The model is trained on real-world Amazon review data and demonstrates the power of machine learning in understanding human sentiment.

---

## 📦 Dataset

**Source:** [Kaggle – Amazon Product Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)  
**Format:** CSV with columns including review text, rating, product info, etc.

---

## 🧠 Objectives

- Preprocess and clean raw review text
- Label reviews based on numerical star ratings
- Vectorize using **TF-IDF**
- Train a classifier using **Logistic Regression**
- Evaluate performance with precision, recall, and confusion matrix

---

## 📊 Features Extracted

- `review` – actual customer text
- `rating` – numerical star rating (1 to 5)
- `label` – derived sentiment:
  - 1–2 stars → `Negative`
  - 3 stars → `Neutral`
  - 4–5 stars → `Positive`

---

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- NLTK for text preprocessing
- Scikit-learn (TF-IDF, train-test split, Logistic Regression)
- Matplotlib / Seaborn (optional for visualization)
- Joblib (for model saving)

---

## ⚙️ How to Run
```
### 1️⃣ Install Required Libraries
pip install pandas scikit-learn nltk joblib
### 2️⃣ Run the Notebook or Script
python sentiment_analysis.py  # or run .ipynb in Colab
### 3️⃣ Output
Console will display model metrics
Model + Vectorizer saved as .pkl files
```
## 📈 Model Performance
Achieves strong performance on balanced datasets with clear polarity. Works well in basic NLP pipelines for product feedback monitoring.
```
Precision    Recall    F1-Score
Positive     0.89       0.91
Negative     0.87       0.84
Neutral      0.78       0.74
```
## 📁 Project Structure
sentiment-analysis/
├── data/
│   └── product_reviews.csv
├── notebooks/
│   └── 01_sentiment_model.ipynb
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── README.md

## 🚀 Future Improvements
	•	Add deep learning models (e.g., LSTM or BERT)
	•	Streamlit or Flask app for user-facing prediction
	•	Real-time sentiment scoring API for products

## 🔗 Author

👤 Kanish Tyagi
📫 kanishtyagi123@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/kanishtyagi123) | [GitHub](https://github.com/kanish5)


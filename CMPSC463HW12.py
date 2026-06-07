import kagglehub
import pandas as pd
import numpy as np
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords

# -----------------------------
# Download dataset
# -----------------------------
path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
print("Path:", path)

# Load train file
file_path = os.path.join(path, "train.csv")
df = pd.read_csv(file_path)

# Columns: [class, title, description]
df.columns = ["label", "title", "description"]

# Combine text
df["text"] = df["title"] + " " + df["description"]

# -----------------------------
# Sample data
# -----------------------------
df_sample = df.sample(n=6000, random_state=42)

train_df = df_sample.iloc[:5000]
test_df = df_sample.iloc[5000:]

# -----------------------------
# Preprocessing
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

X_train = train_df["text"].apply(preprocess)
X_test = test_df["text"].apply(preprocess)

y_train = train_df["label"]
y_test = test_df["label"]

# -----------------------------
# 1. Logistic Regression (BoW)
# -----------------------------
bow = CountVectorizer(max_features=5000)
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_bow, y_train)

pred_lr = lr.predict(X_test_bow)
acc_lr = accuracy_score(y_test, pred_lr)

# -----------------------------
# 2. SVM (TF-IDF)
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

pred_svm = svm.predict(X_test_tfidf)
acc_svm = accuracy_score(y_test, pred_svm)

# -----------------------------
# Results
# -----------------------------
print("\nLogistic Regression (BoW) Accuracy:", acc_lr)
print("\nSVM (TF-IDF) Accuracy:", acc_svm)

print("\nLR Report:\n", classification_report(y_test, pred_lr))
print("\nSVM Report:\n", classification_report(y_test, pred_svm))
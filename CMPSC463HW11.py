import kagglehub
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download dataset
# -----------------------------
path = kagglehub.dataset_download("sahilkirpekar/bbcnews-dataset")
print("Path:", path)

file_path = os.path.join(path, "BBCNews.csv")
df = pd.read_csv(file_path)

# -----------------------------
# Fix dataset columns
# -----------------------------
df = df.drop(columns=['Unnamed: 0'])
texts = df['descr']   # <-- FIXED

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# 1. Text Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

processed_texts = texts.apply(preprocess)

# -----------------------------
# 2. Feature Representation
# -----------------------------
bow_vectorizer = CountVectorizer(max_features=1000)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

X_bow = bow_vectorizer.fit_transform(processed_texts)
X_tfidf = tfidf_vectorizer.fit_transform(processed_texts)

# -----------------------------
# 3. Top 20 Keywords
# -----------------------------
def get_top_keywords(vectorizer, X, n=20):
    sums = np.array(X.sum(axis=0)).flatten()
    words = np.array(vectorizer.get_feature_names_out())
    top_idx = sums.argsort()[-n:][::-1]
    return words[top_idx]

top_bow = get_top_keywords(bow_vectorizer, X_bow)
top_tfidf = get_top_keywords(tfidf_vectorizer, X_tfidf)

print("\nTop 20 BoW Keywords:\n", top_bow)
print("\nTop 20 TF-IDF Keywords:\n", top_tfidf)

# -----------------------------
# 4. Clustering (K-Means)
# -----------------------------
k = 5

kmeans_bow = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_tfidf = KMeans(n_clusters=k, random_state=42, n_init=10)

labels_bow = kmeans_bow.fit_predict(X_bow)
labels_tfidf = kmeans_tfidf.fit_predict(X_tfidf)

# -----------------------------
# 5. Evaluation
# -----------------------------
sil_bow = silhouette_score(X_bow, labels_bow)
sil_tfidf = silhouette_score(X_tfidf, labels_tfidf)

print("\nSilhouette Score (BoW):", sil_bow)
print("Silhouette Score (TF-IDF):", sil_tfidf)

# -----------------------------
# Optional: Visualization
# -----------------------------
plt.bar(["BoW", "TF-IDF"], [sil_bow, sil_tfidf])
plt.title("Clustering Performance Comparison")
plt.ylabel("Silhouette Score")
plt.show()
import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
data = fetch_20newsgroups(subset='all')

# pick 10 categories
random.seed(42)
categories = random.sample(list(data.target_names), 10)

dataset = fetch_20newsgroups(subset='all', categories=categories)

X = np.array(dataset.data)
y = np.array(dataset.target)

# sampling (fixes crash)
train_texts, test_texts = [], []
train_labels, test_labels = [], []

for label in np.unique(y):
    idx = np.where(y == label)[0]
    np.random.shuffle(idx)

    split = int(0.8 * len(idx))  # 80% train, 20% test

    train_idx = idx[:split]
    test_idx = idx[split:]

    train_texts.extend(X[train_idx])
    test_texts.extend(X[test_idx])
    train_labels.extend(y[train_idx])
    test_labels.extend(y[test_idx])

# Bag of Words (smaller)
bow = CountVectorizer(stop_words='english', max_features=3000)
X_train_bow = bow.fit_transform(train_texts)
X_test_bow = bow.transform(test_texts)

model_bow = MLPClassifier(
    hidden_layer_sizes=(100,),
    learning_rate_init=0.001,
    alpha=0.0001,
    max_iter=50,   # smaller for speed
    random_state=42
)

model_bow.fit(X_train_bow, train_labels)
pred_bow = model_bow.predict(X_test_bow)

print("BoW Accuracy:", accuracy_score(test_labels, pred_bow))

# 4. TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf = tfidf.transform(test_texts)

model_tfidf = MLPClassifier(
    hidden_layer_sizes=(100,),
    learning_rate_init=0.001,
    alpha=0.0001,
    max_iter=10,
    random_state=42
)

model_tfidf.fit(X_train_tfidf, train_labels)
pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("TF-IDF Accuracy:", accuracy_score(test_labels, pred_tfidf))
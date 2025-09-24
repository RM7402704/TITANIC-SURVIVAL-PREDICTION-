#train.py
# train.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Small training dataset
texts = [
    "I love this phone", 
    "This product is amazing",
    "Very happy with the quality",
    "Worst purchase ever",
    "I hate this",
    "Terrible experience",
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative"
]

# Convert text to numbers (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X, labels)

# Save the model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved successfully!")

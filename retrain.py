import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv(r"C:\Users\MATHUMITHA\Downloads\archive (3)\Fake.csv")
true = pd.read_csv(r"C:\Users\MATHUMITHA\Downloads\archive (3)\True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Use the text column
X = data["text"]
y = data["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(tfidf_train, y_train)

# Accuracy
y_pred = model.predict(tfidf_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully.")
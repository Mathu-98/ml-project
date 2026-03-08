import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# News example
news = ["The government announced a new economic policy to support businesses."]

# Convert to vector
vector = vectorizer.transform(news)

# Predict
prediction = model.predict(vector)

# Output
if prediction[0] == 1:
    print("This is Real News")
else:
    print("This is Fake News")
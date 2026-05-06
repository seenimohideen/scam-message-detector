import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Simple dataset
messages = [
    "Win money now",
    "Click this link",
    "Urgent account verify",
    "Free prize available",
    "Hello how are you",
    "Let's meet tomorrow",
    "Call me later",
    "Good morning"
]

labels = [1, 1, 1, 1, 0, 0, 0, 0]

# Convert to DataFrame
df = pd.DataFrame({
    "message": messages,
    "label": labels
})

# Text → numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])

# Train model
model = MultinomialNB()
model.fit(X, df["label"])

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model created successfully!")

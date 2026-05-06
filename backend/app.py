# -----------------------------
# 📦 IMPORT LIBRARIES
# -----------------------------
from flask import Flask, request, jsonify      
from flask_cors import CORS                   
from sklearn.feature_extraction.text import TfidfVectorizer  # Text → numbers
from sklearn.naive_bayes import MultinomialNB              # ML model



app = Flask(__name__)
CORS(app)   # Enable frontend access



# TRAINING DATA

messages = [
    "You won lottery click here to claim",
    "Your bank account is blocked send OTP",
    "Congratulations you won cash prize",
    "Click this link to get free money",
    "Meeting at 5 pm today",
    "Can we go for dinner tonight",
    "Please call me when you are free",
    "Project submission is tomorrow"
]


labels = [
    1,  # scam
    1,  # scam
    1, 1,  # scam
    0,  # safe
    0,  # safe
    0,  # safe
    0   # safe
]



# STEP 1: TEXT → NUMBERS

vectorizer = TfidfVectorizer()

# Learn words + convert to numbers
X = vectorizer.fit_transform(messages)



#STEP 2: TRAIN MODEL

model = MultinomialNB()

model.fit(X, labels)



#  API ENDPOINT

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from frontend
    data = request.get_json()

    # Extract message
    message = data['message']

    # Convert message → numbers
    input_vector = vectorizer.transform([message])

    # Predict result (0 or 1)
    prediction = model.predict(input_vector)[0]

    # Get confidence score
    confidence = model.predict_proba(input_vector)[0][prediction]

    # Convert to readable output
    result = "SCAM" if prediction == 1 else "SAFE"

    # Send response
    return jsonify({
        "message": message,
        "result": result,
        "confidence": round(float(confidence), 2)
    })



#RUN SERVER

if __name__ == '__main__':
    app.run(debug=True)
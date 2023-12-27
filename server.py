import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('ScamRakshak.pkl', 'rb') as combined_file:
    model_and_vectorizer = pickle.load(combined_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_text = data['text']

    new_text_features = model_and_vectorizer['vectorizer'].transform([new_text])
    
    prediction = model_and_vectorizer['model'].predict(new_text_features)[0]

    # Prepare the response
    response = {'prediction': 'scam' if prediction == 0 else 'not_scam'}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Or any desired host and port

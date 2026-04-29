from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd
import joblib

app = Flask(__name__)

# Load artifacts
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Predict Route
@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess input
        transformed_data = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(transformed_data)[0]
        probability = model.predict_proba(transformed_data)[0][1]
        
        result = "Customer will churn" if prediction == 1 else "Customer will not churn"
        
        return render_template('index.html',
                               prediction_test = result,
                               probability=round(probability,2))
    
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
    
        
        



from flask import Flask, request, render_template
import requests


app = Flask(__name__)


# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = request.form.to_dict()
    payload = {
        "Gender": form_data["Gender"],
        "Senior_Citizen": form_data["Senior Citizen"],
        "Partner": form_data["Partner"],
        "Dependents": form_data["Dependents"],
        "Tenure_Months": int(form_data["Tenure Months"]),
        "Phone_Service": form_data["Phone Service"],
        "Multiple_Lines": form_data["Multiple Lines"],
        "Internet_Service": form_data["Internet Service"],
        "Online_Security": form_data["Online Security"],
        "Online_Backup": form_data["Online Backup"],
        "Device_Protection": form_data["Device Protection"],
        "Tech_Support": form_data["Tech Support"],
        "Streaming_TV": form_data["Streaming TV"],
        "Streaming_Movies": form_data["Streaming Movies"],
        "Contract": form_data["Contract"],
        "Paperless_Billing": form_data["Paperless Billing"],
        "Payment_Method": form_data["Payment Method"],
        "Monthly_Charges": float(form_data["Monthly Charges"]),
        "Total_Charges": float(form_data["Total Charges"]),
        "CLTV": int(form_data["CLTV"])
    }

    response = requests.post(
        "http://fastapi-app:8000/predict",
        json=payload)
    
    if response.status_code == 200:
        prediction_result = response.json()

        return render_template(
        'index.html',
        prediction_text=prediction_result["prediction"],
        probability=prediction_result["probability"]
      )

    else:
        return f"FastAPI Error: {response.text}"

if __name__ == "__main__":
    app.run(debug=True)
    
        
        



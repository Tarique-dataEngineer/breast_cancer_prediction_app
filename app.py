from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('svm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    age = float(data['age'])
    bmi = float(data['bmi'])
    glucose = float(data['glucose'])
    insulin = float(data['insulin'])
    homa = float(data['homa'])
    leptin = float(data['leptin'])
    adiponectin = float(data['adiponectin'])
    resistin = float(data['resistin'])
    mcp1 = float(data['mcp1'])

    # Prepare the input data for prediction
    input_data = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]])

    # Make prediction
    prediction = model.predict(input_data)

    # Determine the result message based on the prediction
    if prediction == 1:
        result_message = "You are healthy. You don't have breast cancer."
    else:
        result_message = "You may have breast cancer. Please consult a doctor immediately."

    return render_template('index.html', prediction_text=result_message)

if __name__ == "__main__":
    app.run(debug=True)
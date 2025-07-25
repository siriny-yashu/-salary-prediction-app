from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = int(request.form['experience'])
    education = int(request.form['education'])

    prediction = model.predict([[experience, education]])
    output = round(prediction[0], 2)

    return f"<h3>Predicted Salary: ₹{output}</h3><a href='/'>Go Back</a>"

if __name__ == "__main__":
    app.run(debug=True)

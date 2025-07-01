from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('lr_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = int(request.form['area'])
        age = int(request.form['age'])
        stairs = int(request.form['stairs'])

        input_data = np.array([[area, age, stairs]])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f'Estimated House Price: ₹{prediction} lakhs')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True,port=1200)

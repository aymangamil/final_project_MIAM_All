##Flask Application
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# --- 1. Load the Pre-trained Model ---
try:
    with open("C:\\Users\\Ayman\\Downloads\\MIAM\\Final_project\\logistic_model_Random_feature_select.pkl", 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found. Please check the file path.")
    model = None


@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded!"

    try:
        school = request.form['school']
        reason = request.form['reason']
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        schoolsup = request.form['schoolsup']
        higher = request.form['higher']
        subject = request.form['subject']
        medu = int(request.form['medu'])
        fedu = int(request.form['fedu'])
        dalc = int(request.form['dalc'])
        walc = int(request.form['walc'])
        g1 = int(request.form['g1'])
        g2 = int(request.form['g2'])

        user_data = {
            'school': 'GP' if school == "Gabriel Pereira (GP)" else 'MS',
            'Medu': medu,
            'Fedu': fedu,
            'reason': reason,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': 'yes' if schoolsup == "Yes" else 'no',
            'higher': 'yes' if higher == "Yes" else 'no',
            'Dalc': dalc,
            'Walc': walc,
            'G1': g1,
            'G2': g2,
            'Subject': 'math' if subject == "Mathematics" else 'portuguese',
        }

        df = pd.DataFrame([user_data])

        df['school'] = df['school'].map({'GP': 1, 'MS': 0})
        df['reason'] = df['reason'].map({'home': 1, 'reputation': 2, 'course': 3, 'other': 4})
        df['schoolsup'] = df['schoolsup'].map({'yes': 1, 'no': 0})
        df['higher'] = df['higher'].map({'yes': 1, 'no': 0})
        df['Subject'] = df['Subject'].map({'math': 1, 'portuguese': 0})
        df['Averge_rate'] = (df['G1'] + df['G2']) / 2

        features = ['school', 'Medu', 'Fedu', 'reason', 'studytime', 'failures', 
                    'schoolsup', 'higher', 'Dalc', 'Walc', 'G1', 'G2', 'Subject', 'Averge_rate']
        processed_data = df[features]

        prediction = model.predict(processed_data)[0]

        return render_template("index.html", prediction_text=f"Predicted Final Grade: {prediction}")

    except Exception as e:
        return f"Error during prediction: {e}"


if __name__ == '__main__':
    app.run(debug=True)


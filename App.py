import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set the title of the Streamlit app
st.title("Student Final Grade Predictor")
st.write("Enter the student's information below to predict their final grade.")

# --- 1. Load the Pre-trained Model ---
# Use a try-except block to handle potential file errors
try:
    with open("C:\\Users\\Ayman\\Downloads\\MIAM\\Final_project\\logistic_model_Random_feature_select.pkl", 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: The model file was not found. Please check the file path.")
    st.stop() # Stop the app if the model cannot be loaded

# --- 2. Create the User Interface (UI) with Streamlit Widgets ---

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Student Details")
    school = st.selectbox("School", ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"])
    reason = st.selectbox("Reason for choosing school", ["home", "reputation", "course", "other"])
    studytime = st.slider("Weekly study time (hours)", 1, 4, 2)
    failures = st.slider("Number of past class failures", 0, 4, 0)
    schoolsup = st.radio("Has extra educational support?", ["Yes", "No"])
    higher = st.radio("Wants to take higher education?", ["Yes", "No"])
    subject = st.radio("Subject", ["Mathematics", "Portuguese"])

with col2:
    st.subheader("Parent and Lifestyle Details")
    medu = st.slider("Mother's education level", 0, 4, 2)
    fedu = st.slider("Father's education level", 0, 4, 2)
    dalc = st.slider("Workday alcohol consumption", 1, 5, 1)
    walc = st.slider("Weekend alcohol consumption", 1, 5, 1)
    g1 = st.slider("First period grade (0-20)", 0, 20, 10)
    g2 = st.slider("Second period grade (0-20)", 0, 20, 10)

# --- 3. Preprocess the User Inputs for the Model ---

# Create a button to trigger the prediction
if st.button("Predict Final Grade"):
    
    # Create a dictionary from the Streamlit inputs
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
    
    # Create a DataFrame from the dictionary
    df = pd.DataFrame([user_data])
    
    # Map categorical features to numerical values
    df['school'] = df['school'].map({'GP': 1, 'MS': 0})
    df['reason'] = df['reason'].map({'home': 1, 'reputation': 2, 'course': 3, 'other': 4})
    df['schoolsup'] = df['schoolsup'].map({'yes': 1, 'no': 0})
    df['higher'] = df['higher'].map({'yes': 1, 'no': 0})
    df['Subject'] = df['Subject'].map({'math': 1, 'portuguese': 0})
    
    # Calculate 'Average_rate'
    df['Averge_rate'] = (df['G1'] + df['G2']) / 2
    
    # Reorder columns to match the model's training data
    features = ['school', 'Medu', 'Fedu', 'reason', 'studytime', 'failures', 'schoolsup', 'higher', 'Dalc', 'Walc', 'G1', 'G2', 'Subject', 'Averge_rate']
    processed_data = df[features]
    
    # --- 4. Make the Prediction and Display the Result ---
    
    try:
        prediction = model.predict(processed_data)[0]
        st.subheader("Prediction Result")
        st.success(prediction)
        
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
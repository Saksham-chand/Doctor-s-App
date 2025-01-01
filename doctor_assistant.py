import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load models
try:
    diabetes_model = load_model('diabetes_model.h5')
    brain_tumor_model = load_model('Brain_Tumor.keras')
    skin_cancer_model = load_model('Skin_Cancer_model.keras')
except Exception as e:
    st.error(f"Error loading models: {e}")

# Define the app
st.title("Doctor's Assistant App")
st.sidebar.header("Select Disease")

# Sidebar for disease selection
disease = st.sidebar.selectbox(
    "Choose the disease you want to predict:",
    ("Diabetes", "Brain Tumor Detection", "Skin Cancer Detection")
)

# Function for diabetes prediction
def predict_diabetes(inputs):
    try:
        # Convert the list to a NumPy array and reshape it to match the model's expected input shape
        inputs_array = np.array(inputs).reshape(1, -1)  # Reshaping to (1, number of features)
        # Predict with the model
        prediction = diabetes_model.predict(inputs_array)
        # Return the prediction result
        return "Diabetic" if prediction[0] > 0.5 else "Not Diabetic"
    except Exception as e:
        st.error(f"Error in diabetes prediction: {e}")
        return None

# Function for brain tumor detection
def predict_brain_tumor(image):
    try:
        img = image.resize((224, 224))  # Ensure size matches model input
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = brain_tumor_model.predict(img_array)
        return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
    except Exception as e:
        st.error(f"Error in brain tumor prediction: {e}")
        return None

# Function for skin cancer detection
def predict_skin_cancer(image):
    try:
        img = image.resize((224, 224))  # Ensure size matches model input
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = skin_cancer_model.predict(img_array)
        classes = ["Actinic Keratoses and Intraepithelial Carcinoma", "Basal Cell carcinoma", "Benign Keratosis", "Dermatofibroma","Melanoma (Melign)",
                   "Melanocytic Nevi","Vacular lesions"]  # Replace with actual classes
        return classes[np.argmax(prediction)]
    except Exception as e:
        st.error(f"Error in skin cancer prediction: {e}")
        return None

# Main logic for diabetes prediction
if disease == "Diabetes":
    st.header("Diabetes Prediction")
    
    # Input fields for diabetes features
    pregnancies = st.number_input("Pregnancies", 0)
    glucose = st.number_input("Glucose Level", 0)
    blood_pressure = st.number_input("Blood Pressure", 0)
    skin_thickness = st.number_input("Skin Thickness", 0)
    insulin = st.number_input("Insulin Level", 0)
    bmi = st.number_input("BMI", 0.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 0.0)
    age = st.number_input("Age", 0)

    # Button to trigger prediction
    if st.button("Predict"):
        # Prepare input list for prediction
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        
        # Make prediction and display result
        result = predict_diabetes(input_data)
        st.success(f"Prediction: {result}")

elif disease == "Brain Tumor Detection":
    st.header("Brain Tumor Detection")
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            result = predict_brain_tumor(image)
            if result:
                st.success(f"Prediction: {result}")

elif disease == "Skin Cancer Detection":
    st.header("Skin Cancer Detection")
    uploaded_file = st.file_uploader("Upload a Skin Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            result = predict_skin_cancer(image)
            if result:
                st.success(f"Prediction: {result}")

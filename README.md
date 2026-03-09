🩺 Doctor’s App – AI Powered Medical Diagnostic System

An AI-powered healthcare diagnostic application built using Streamlit and Deep Learning that assists in early detection of medical conditions using machine learning models.

The application integrates multiple medical diagnostic models to analyze patient data and medical images, helping provide quick and accessible insights for healthcare assistance.

This project demonstrates the practical use of Artificial Intelligence in healthcare through disease prediction and medical image classification.

🚀 Features
🧠 Brain Tumor Detection

Detects the presence of brain tumors using MRI images

Deep learning CNN model trained on 4000+ MRI scans

Classifies images into Healthy or Tumor

Achieved 98.12% accuracy

🩸 Diabetes Prediction

Predicts the likelihood of diabetes

Uses Linear Regression and CNN models

Trained on 1000+ medical data samples

Achieved 90.12% accuracy

🔬 Skin Cancer Classification

Classifies 7 different types of skin cancer

Uses pre-trained deep learning models

VGG16

MobileNetV2

Trained on 30,000+ dermatology images

Achieved 88.68% accuracy

🧠 Technologies Used
Category	Technologies

Programming	Python 

Framework	Streamlit

Machine Learning	Scikit-learn

Deep Learning	TensorFlow / Keras

Image Processing	OpenCV

Data Analysis	NumPy, Pandas

Visualization	Matplotlib

🏗️ System Architecture
User Input
     |
     ▼
     
Streamlit Interface
     │
     ▼
     
AI Models
 ├── Diabetes Prediction Model
 ├── Brain Tumor Detection Model
 └── Skin Cancer Classification Model
     │
     ▼
Prediction Result Display
📂 Project Structure
Doctor-s-App
│
├── models/
│   ├── diabetes_model.pkl
│   ├── brain_tumor_model.h5
│   └── skin_cancer_model.h5
│
├── datasets/
│
├── app.py
├── requirements.txt
└── README.md
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Saksham-chand/Doctor-s-App.git
cd Doctor-s-App
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run app.py
📊 Model Performance
Model	Dataset Size	Accuracy
Diabetes Prediction	1000+ records	90.12%
Brain Tumor Detection	4000 MRI images	98.12%
Skin Cancer Classification	30,000 images	88.68%
📈 Future Improvements

Integration with real hospital datasets

Doctor consultation integration

Cloud deployment

Mobile healthcare app

More disease prediction models

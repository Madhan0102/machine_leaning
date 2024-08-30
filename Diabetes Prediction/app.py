import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

background_image ="https://editor.analyticsvidhya.com/uploads/30738medtec-futuristic-650.jpg"

# Custom CSS to add the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({background_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Image slider
image_slider = """
<style>
.slider {
    width: 100%;
    max-width: 800px;
    margin: auto;
    overflow: hidden;
    border: 2px solid #ddd;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
}

.slides {
    display: flex;
    transition: transform 0.5s ease-in-out;
    animation: slide 16s infinite;
}

.slides img {
    width: 100%;
    border: 2px solid #fff;
}

@keyframes slide {
    0% { transform: translateX(0%); }
    20% { transform: translateX(0%); }
    25% { transform: translateX(-100%); }
    45% { transform: translateX(-100%); }
    50% { transform: translateX(-200%); }
    70% { transform: translateX(-200%); }
    75% { transform: translateX(-300%); }
    95% { transform: translateX(-300%); }
    100% { transform: translateX(0%); }
}
</style>

<div class="slider">
    <div class="slides">
        <img src="https://slidemodel.com/wp-content/uploads/60607-01-diabetes-powerpoint-template-16x9-14.jpg" alt="Image 1">
        <img src="https://slidemodel.com/wp-content/uploads/60607-01-diabetes-powerpoint-template-16x9-14.jpg" alt="Image 2">
        <img src="https://slidemodel.com/wp-content/uploads/60607-01-diabetes-powerpoint-template-16x9-14.jpg" alt="Image 3">
    </div>
</div>
"""

# Display the image slider
st.markdown(image_slider, unsafe_allow_html=True)

# Load the machine learning model
with open(r'C:\Users\conne\OneDrive\Documents\GitHub\machine_leaning\Diabetes Prediction\trained_model.sav','rb') as file:
    model = pickle.load(file)


# Title of the application
st.title("Diabetes Prediction")
# Create sliders and input fields for each feature
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=140, value=70)
SkinThickness = st.number_input('SkinThickness', min_value=0, max_value=100, value=20)
Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input('Age', min_value=0, max_value=120, value=30)

# Store the inputs into a dataframe
input_data = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [Age]
})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    st.write(f"The predicted outcome is: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    
    # Plot the prediction (binary output)
    fig, ax = plt.subplots()
    ax.bar(['No Diabetes', 'Diabetes'], [1 - prediction, prediction])
    ax.set_ylabel('Probability')
    st.pyplot(fig)
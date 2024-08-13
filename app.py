import streamlit as st
# import pickle
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained pipeline
# pipe = pickle.load(open('pipe.pkl', 'rb'))
# Load the pre-trained pipeline using joblib
pipeline_rf = joblib.load('model_pipeline.joblib')

def main():
    # Custom CSS for styling the title
    title_style = """
    <style>
        /* Custom CSS for the title */
        .title-text {
            color: #0874D1; /* Change text color to blue */
            font-size: 36px; /* Increase font size */
            font-weight: bold; /* Make text bold */
            text-align: left; /* Align text to left */
            margin-bottom: 20px; /* Add some bottom margin */
            margin-top: 0px;
        }
    </style>
    """
    # Display the styled title with emoji
    st.markdown(title_style, unsafe_allow_html=True)
    st.markdown("<h1 class='title-text'>🩺 Diabetes Risk Prediction</h1>", unsafe_allow_html=True)

    st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 400px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Set input field width
    st.markdown(
    """
    <style>
    /* Input field width */
    div.stTextInput > div:first-child input[type="text"],
    div.stNumberInput > div:first-child input[type="number"],
    div.stSelectbox > div:first-child select {
        width: 300px;
        height: 40px;
        border-radius: 8px;
        border: 2px solid #1E88E5;
        padding: 5px 10px;
        font-size: 16px;
        color: #1E88E5;
        background-color: #F5F5F5;
    }

    /* Increase spacing between input fields */
    .stTextInput, .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }

    /* Style the select box */
    div.stSelectbox > div:first-child::after {
        color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

    # Inputs for diabetes risk prediction
    age = st.number_input('Age', min_value=0, max_value=100, value=25, step=1)
    gender = st.radio('Gender', ['Male', 'Female'], index=0, horizontal=True)
    polyuria = st.radio('Polyuria (Excessive urination)', ['Yes', 'No'], index=1, horizontal=True)
    polydipsia = st.radio('Polydipsia (Excessive thirst)', ['Yes', 'No'], index=1, horizontal=True)
    sudden_weight_loss = st.radio('Sudden Weight Loss (Unexplained weight loss)', ['Yes', 'No'], index=1, horizontal=True)
    weakness = st.radio('Weakness (General fatigue)', ['Yes', 'No'], index=1, horizontal=True)
    polyphagia = st.radio('Polyphagia (Excessive hunger)', ['Yes', 'No'], index=1, horizontal=True)
    genital_thrush = st.radio('Genital Thrush (Yeast infection)', ['Yes', 'No'], index=1, horizontal=True)
    visual_blurring = st.radio('Visual Blurring (Blurred vision)', ['Yes', 'No'], index=1, horizontal=True)
    itching = st.radio('Itching (Persistent itching)', ['Yes', 'No'], index=1, horizontal=True)
    irritability = st.radio('Irritability (Mood swings)', ['Yes', 'No'], index=1, horizontal=True)
    delayed_healing = st.radio('Delayed Healing (Slow wound healing)', ['Yes', 'No'], index=1, horizontal=True)
    partial_paresis = st.radio('Partial Paresis (Muscle weakness or paralysis)', ['Yes', 'No'], index=1, horizontal=True)
    muscle_stiffness = st.radio('Muscle Stiffness (Stiff muscles)', ['Yes', 'No'], index=1, horizontal=True)
    alopecia = st.radio('Alopecia (Hair loss)', ['Yes', 'No'], index=1, horizontal=True)
    obesity = st.radio('Obesity (High body fat)', ['Yes', 'No'], index=1, horizontal=True)

    # Predict button
    if st.button('Predict'):
        # Convert categorical variables to numeric
        gender = 1 if gender == 'Male' else 0
        polyuria = 1 if polyuria == 'Yes' else 0
        polydipsia = 1 if polydipsia == 'Yes' else 0
        sudden_weight_loss = 1 if sudden_weight_loss == 'Yes' else 0
        weakness = 1 if weakness == 'Yes' else 0
        polyphagia = 1 if polyphagia == 'Yes' else 0
        genital_thrush = 1 if genital_thrush == 'Yes' else 0
        visual_blurring = 1 if visual_blurring == 'Yes' else 0
        itching = 1 if itching == 'Yes' else 0
        irritability = 1 if irritability == 'Yes' else 0
        delayed_healing = 1 if delayed_healing == 'Yes' else 0
        partial_paresis = 1 if partial_paresis == 'Yes' else 0
        muscle_stiffness = 1 if muscle_stiffness == 'Yes' else 0
        alopecia = 1 if alopecia == 'Yes' else 0
        obesity = 1 if obesity == 'Yes' else 0

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Polyuria': [polyuria],
            'Polydipsia': [polydipsia],
            'sudden weight loss': [sudden_weight_loss],
            'weakness': [weakness],
            'Polyphagia': [polyphagia],
            'Genital thrush': [genital_thrush],
            'visual blurring': [visual_blurring],
            'Itching': [itching],
            'Irritability': [irritability],
            'delayed healing': [delayed_healing],
            'partial paresis': [partial_paresis],
            'muscle stiffness': [muscle_stiffness],
            'Alopecia': [alopecia],
            'Obesity': [obesity]
        })

        if input_data.isnull().values.any():
            st.error("Input contains missing values. Please fill all the fields.")
            return


        try:
            # Make prediction using the pre-trained pipeline
            prediction = pipeline_rf.predict(input_data)[0]

            # Display the prediction result
            if prediction == 1:
                st.title('Prediction: High Risk of Diabetes')
                st.success("You are at high risk of diabetes. Please consult a healthcare provider for further assessment.")
            else:
                st.title('Prediction: Low Risk of Diabetes')
                st.success("You are at low risk of diabetes. Maintain a healthy lifestyle and regular check-ups.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

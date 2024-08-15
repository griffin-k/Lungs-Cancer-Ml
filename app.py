import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# Load the saved model and selected features
with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)
    lung_cancer_model = model_data['model']
    selected_features = model_data['selected_features']

st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

st.title("Lung Cancer Detection Web App")
st.sidebar.title("Predicting Lung Cancer using ML models")
with st.sidebar:
    selected = option_menu('Lung Cancer Prediction System',
                           ['Lung Cancer Detection',
                            'About Lung Cancer Detection Project',
                            'Binary Classification Categories',
                            'About Us'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'lungs-fill', 'person'],
                           default_index=0)

def get_user_input():
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.slider("Age", 30, 100, 50)  # Add age input
        smoking = st.selectbox("Smoking", ["1", "2"])  # Match encoding in your data
        yellow_fingers = st.selectbox("Yellow Fingers", ["1", "2"])
        anxiety = st.selectbox("Anxiety", ["1", "2"])
    with col2:
        peer_pressure = st.selectbox("Peer Pressure", ["1", "2"])
        chronic_disease = st.selectbox("Chronic Disease", ["1", "2"])
        fatigue = st.selectbox("Fatigue", ["1", "2"])
        allergy = st.selectbox("Allergy", ["1", "2"])
        wheezing = st.selectbox("Wheezing", ["1", "2"])
    with col3:
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["1", "2"])
        coughing = st.selectbox("Coughing", ["1", "2"])
        shortness_of_breath = st.selectbox("Shortness of Breath", ["1", "2"])
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["1", "2"])
        chest_pain = st.selectbox("Chest Pain", ["1", "2"])

    features_list = [
        gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
        chronic_disease, fatigue, allergy, wheezing, alcohol_consumption,
        coughing, shortness_of_breath, swallowing_difficulty, chest_pain
    ]
    return np.array(features_list, dtype=object)

# Function to predict cancer risk
def predict_cancer_risk(features):
    if features is not None:
        # Convert categorical variables
        features[0] = 1 if features[0] == "M" else 0  # Convert gender to binary
        features[1:] = [int(feature) for feature in features[1:]]  # Convert all features to integers

        # Ensure the features match the ones used for training
        features = features[selected_features]
        features = features.reshape(1, -1)

        if st.button('Lung Cancer Prediction Result'):
            with st.spinner('Processing your prediction...'):
                cancer_prob = lung_cancer_model.predict_proba(features)
                st.write(f"Prediction probabilities: {cancer_prob}")  # Check prediction probabilities

                cancer_pred = lung_cancer_model.predict(features)
                st.write(f"Raw prediction: {cancer_pred}")  # Debugging line

                # Interpret the model's output
                if cancer_pred[0] == 1:
                    cancer_diagnosis = "Lung cancer is diagnosed."
                else:
                    cancer_diagnosis = "Lung cancer is not diagnosed."

                st.success(f"The prediction is: {cancer_diagnosis}")


# Display pages based on sidebar selection
if selected == "Lung Cancer Detection":
    features = get_user_input()
    if features.any():
        predict_cancer_risk(features)

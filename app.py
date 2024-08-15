import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Cancer watch",
                   layout="wide",
                   page_icon="🧑‍⚕️")


working_dir = os.path.dirname(os.path.abspath(__file__))
lung_cancer_model = pickle.load(open(f'{working_dir}/saved_models/lung_cancer_model.sav', 'rb'))


st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f9;
    }

    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .stMarkdown {
        font-size: 16px;
    }

    .stTitle {
        color: #333333;
    }

    .stWrite {
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu('Cancer watch',
                           ['Lung Cancer Prediction', 'Binary Classification Categories', 'Developer Info'],
                           menu_icon='hospital-fill',
                           icons=['lungs', 'list', 'people'],
                           default_index=0)


if selected == 'Lung Cancer Prediction':
    st.title('Lung Cancer Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age')

    with col2:
        Smoking = st.text_input('Smoking History (1 for Yes, 0 for No)')

    with col3:
        Yellow_Fingers = st.text_input('Yellow Fingers (1 for Yes, 0 for No)')

    with col1:
        Anxiety = st.text_input('Anxiety (1 for Yes, 0 for No)')

    with col2:
        Chronic_Disease = st.text_input('Chronic Disease (1 for Yes, 0 for No)')

    with col3:
        Fatigue = st.text_input('Fatigue (1 for Yes, 0 for No)')

    with col1:
        Wheezing = st.text_input('Wheezing (1 for Yes, 0 for No)')

    with col2:
        Shortness_of_Breath = st.text_input('Shortness of Breath (1 for Yes, 0 for No)')

    if st.button('Lung Cancer Test Result'):
        user_input = [Age, Smoking, Yellow_Fingers, Anxiety, Chronic_Disease, Fatigue,
                      Wheezing, Shortness_of_Breath]
        user_input = [float(x) for x in user_input]
        lung_cancer_prediction = lung_cancer_model.predict([user_input])
        if lung_cancer_prediction[0] == 1:
            st.success('The person is likely to have lung cancer')
        else:
            st.success('The person is unlikely to have lung cancer')


if selected == 'Binary Classification Categories':
    st.title('Binary Classification Categories')
    st.write('**Causes of Lung Cancer:**')
    st.write('**Smoking:** Smoking causes lung cancer by introducing harmful chemicals into the lungs, which can damage cells and lead to the development of tumors.')





if selected == 'Developer Info':
    st.title('Developer Info')
    st.write('**ABOUT US**')
    st.write('**Meet the team**')
    st.write('Laraib Masood : Fa20/BSCS/534/Section C')
    st.write('Amina Nawaz : Fa20/BSCS/093/Section C')
    st.write('**Contact Us**')
    st.write('Laraib Masood : Fa20-bscs-534@lgu.edu.pk')
    st.write('Amina Nawaz : Fa20-bscs-093@lgu.edu.pk')

import streamlit as st
from data_utils import load_data

def show():
    st.title("🌡️ Climate Temperature Prediction")
    st.markdown(
    """
    ## Welcome to the Climate Temperature Prediciton Project

    This application helps analyze climate data and predict maximum temperatures based on 
    various environmental and agricultural factors.
    """
    )

    # Project overview
    st.header("📌 Project Overview")
    st.markdown(
    """
    - **Objective**: Predict maximum temperatures based on historical climate and agricultural
    data
    - **Target Variable**: Average Maximum Temperature (`avg_max_temp`)
    - **Features**: Includes temperature metrics, humidity, precipitation, population density, agricultural land area, etc,
    - **Techniques Used**:
        - Exploratory Data Analysis (EDA)
        - Feature Engineering
        - Machine Learning Modeling
        - Natural Language Processing (for text analysis)
    """
    )

    # Data preview
    st.header("🔍 Data Preview")
    data = load_data()
    st.dataframe(data.head())

    st.markdown(
    """
    ### Navigation Guide
    - **Exploratory Data Analysis**: Visualize and understand the data
    - **Feature Engineering**: Create and select important features
    - **Model Training**: Train machine learning modles
    - **Model Evaluation**: Compare model performance
    - **Prediction**: Make new predictions with trained models
    """
    )

    # NLP demonstration
    st.header("💬 NLP Feature Demonstration")
    user_input = st.text_input("Try our NLP feature - enter some text about climate:")
    if user_input:
        from data_utils import analyze_text
        analysis = analyze_text(user_input)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tokens")
            st.write(analysis['tokens'])

            st.subheader("POS tags")
            st.write(analysis['pos_tags'])
        with col2:
            st.subheader('Lemmas')
            st.write(analysis['lemmas'])

            st.subheader('Entities')
            st.write(analysis['entities'])
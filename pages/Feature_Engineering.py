import streamlit as st
import pandas as pd
import numpy as np
from data_utils import load_data

def show():
    st.title("🛠️ Feature Engineering")
    
    # Load data only when this page is accessed
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    
    data = st.session_state.data.copy()
    
    st.header("1. Existing Features")
    st.write(data.columns.tolist())
    
    st.header("2. Create New Features")
    
    # Feature creation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Features")
        if st.checkbox("Add Temperature Range (max - min)"):
            data['temp_range'] = data['avg_max_temp'] - data['avg_min_temp']
        
        if st.checkbox("Add Temperature Anomaly (deviation from mean)"):
            mean_temp = data['avg_mean_temp'].mean()
            data['temp_anomaly'] = data['avg_mean_temp'] - mean_temp
    
    with col2:
        st.subheader("Time Features")
        if st.checkbox("Add Decade Feature"):
            data['decade'] = (data['year'] // 10) * 10
        
        if st.checkbox("Add Year Difference from Reference (2000)"):
            data['years_from_2000'] = data['year'] - 2000
    
    st.subheader("3. Feature Selection")
    st.write("Select features to include in the model:")
    
    # Let user select features
    all_features = [col for col in data.columns if col != 'avg_max_temp']
    selected_features = st.multiselect(
        "Choose features", 
        all_features,
        default=all_features
    )
    
    # Add target back
    selected_features_with_target = selected_features + ['avg_max_temp']
    engineered_data = data[selected_features_with_target]
    
    # Save to session state
    if st.button("Apply Feature Engineering"):
        st.session_state.data = engineered_data
        st.success("Feature engineering applied! The dataset now has:")
        st.write(f"{engineered_data.shape[1]} features, {engineered_data.shape[0]} rows")
        
        st.subheader("Preview of Engineered Data")
        st.dataframe(engineered_data.head())
    
    st.header("4. Feature Importance Analysis")
    st.write("After training a model, you can view feature importance in the Model Evaluation page.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.title("📊 Exploratory Data Analysis")

    if 'data' not in st.session_state:
        st.error("Data not loaded. Please return to the Home page.")
        return
    
    data = st.session_state.data

    # Basic info
    st.header("1. Data Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

        st.subheader("Data Types")
        st.write(data.dtypes)
    
    with col2:
        st.subheader("Missing Values")
        st.write(data.isnull().sum())

        st.subheader("Basic Statistics")
        st.write(data.describe())
    
    # Time series analysis
    st.header("2. Time Series Analysis")
    time_col = st.selectbox("Select variable for time series plot", data.select_dtypes(include = 'number').columns)

    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=data, x='year', y=time_col, ax=ax)
    ax.set_title(f"{time_col} over Time")
    st.pyplot(fig)

    # Distribution plots
    st.header("3. Distribution Analysis")
    dist_col = st.selectbox("Select variable for distribution plot", data.select_dtypes(include = 'number').columns)

    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    sns.histplot(data=data, x=dist_col, kde=True, ax=ax[0])
    sns.boxplot(data=data, y=dist_col, ax=ax[1])
    st.pyplot(fig)

    # Correlation analysis
    st.header("4. Correlation Analysis")

    fig, ax = plt.subplots(figsize = (12,8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Target variable analysis
    st.header("5. Target Variable Analysis")
    st.subheader("Average Maximum Temperature (avg_max_temp)")

    fig, ax = plt.subplots(1,2, figsize=(15,5))
    sns.histplot(data=data, x='avg_max_temp', kde=True, ax=ax[0])
    sns.boxplot(data=data, y="avg_max_temp", ax=ax[1])
    st.pyplot(fig)

    st.subheader("Top Correlated Features with Target")
    corr_with_target = data.corr()['avg_max_temp'].sort_values(ascending=False)
    st.write(corr_with_target)
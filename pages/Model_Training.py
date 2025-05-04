import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_utils import preprocess_data, save_model
import seaborn as sns
import matplotlib.pyplot as plt
from data_utils import load_data

def show():
    st.title("🤖 Model Training")

    # Load data only when this page is accessed
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    
    data = st.session_state.data

    # Model selection
    st.header("1. Select Model")
    model_options = {
        "Random Forest": RandomForestRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge  
    }

    selected_model = st.selectbox("Choose a model to train", list(model_options.keys()))

    # Hyperparameter tuning
    st.header("2. Hyperparameters")
    params = {}

    if selected_model == "Random Forest":
        params['n_estimators'] = st.slider("Number of tress", 10, 200, 100)
        params['max_depth'] = st.slider("Max depth", 1, 20, 10)
        params['random_state'] = 42

    elif selected_model == "Gradient Boosting":
        params['n_estimators'] = st.slider("Number of tress", 10, 200, 100)
        params['learning_rate'] = st.slider("Learning rate", 0.01, 1.0, 0.1)
        params['max_depth'] = st.slider('Max depth', 1, 10, 3)
        params['random_state'] = 42

    elif selected_model == "Ridge Regression":
        params['alpha'] = st.slider("Alpha (Regularization strength)", 0.01, 10.0, 1.0)
        params['solver'] = st.selectbox("Solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        params['random_state'] = 42

    # Train/test split
    st.header("3. Train/Test Split")
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        data, test_size=test_size
    )

    # Train model
    if st.button("Train Model"):
        st.header("4. Training Results")

        with st.spinner(f"Training {selected_model}..."):
            # Initialize model
            model_class = model_options[selected_model]
            model = model_class(**params)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Display results
            st.success("Model trained successfully!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("MSE", f"{mse:.2f}")
            col3.metric("RMSE", f"{rmse:.2f}")
            col4.metric("R² Score", f"{r2:.2f}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
                st.pyplot(fig)
            
            # Save model
            save_model(model, selected_model.lower().replace(" ", "_"))
            st.session_state.trained_model = model
            st.session_state.model_name = selected_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred

            st.success("Model saved successfully! You can now evaluate it in the Model Evaluation page.")

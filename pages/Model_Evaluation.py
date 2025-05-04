import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.title("📈 Model Evaluation")

    # Check if model is trained
    if 'trained_model' not in st.session_state:
        st.warning("No trained model found. Please train a model first.")
        return

    model = st.session_state.trained_model
    model_name = st.session_state.model_name
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    st.header(f"Evaluating {model_name} Model")

    # Metrics
    st.subheader("1. Performance Metrics")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("R² Score", f"{r2:.2f}")

    # Actual vs Predicted plot
    st.subheader("2. Actual vs Predicted Values")

    fig, ax = plt.subplots(figsize = (10,6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)

    # Residual plot
    st.subheader("3. Residual Analysis")

    residuals = y_test - y_pred
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_title("Residual Distribution")
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[1])
    ax[1].axhline(y=0, color='r', linestyle='--')
    ax[1].set_title("Residuals vs Predicted")
    st.pyplot(fig)

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("4. Feature Importance")

        importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
        st.pyplot(fig)

    # Save evaluation results
    if st.button("Save Evaluation Results"):
        evaluation_results = {
            'model': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        st.session_state.evaluation_results = evaluation_results
        st.success("Evaluation results saved!")

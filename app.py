import streamlit as st
from data_utils import load_data

def main():
    st.set_page_config(
        page_title = "Climate Temperature Prediction",
        page_icon = "🌡️",
        layout = 'wide'
    )

    # Load data once and cache it
    data = load_data()

    # Store data in session state for access across pages
    if 'data' not in st.session_state:
        st.session_state.data = data

    # Sidebar Navigation
    with st.sidebar:
        st.title('Navigation')
        page = st.radio(
            "Go to",
            options=[
                "Home",
                "Exploratory Data Analysis", 
                "Feature Engineering",
                "Model Training",
                "Model Evaluation",
                "Prediction",
                "Climate Text Analysis"
            ],
            key="nav_radio"  
        ) 

    # Display the selected page
    if page == "Home":
        from pages import Home
        st.session_state.current_page = "Home"
        Home.show()
    elif page == "Exploratory Data Analysis":
        from pages import EDA
        st.session_state.current_page = "EDA"
        EDA.show()
    elif page == "Feature Engineering":
        from pages import Feature_Engineering
        st.session_state.current_page = "Feature Engineering"
        Feature_Engineering.show()
    elif page == "Model Training":
        from pages import Model_Training
        st.session_state.current_page = "Model Training"
        Model_Training.show()
    elif page == "Model Evaluation":
        from pages import Model_Evaluation
        st.session_state.current_page = "Model Evaluation"
        Model_Evaluation.show()
    elif page == "Prediction":
        from pages import Prediction
        st.session_state.current_page = "Prediction"
        Prediction.show()
    elif page == "Climate Text Analysis":
        from pages import Climate_Text_Analysis
        st.session_state.current_page = "Climate Text Analysis"
        Climate_Text_Analysis.show()
    
if __name__ == "__main__":
    main()
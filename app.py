import streamlit as st
from data_utils import load_data
from pages import Home, EDA, Feature_Engineering, Model_Evaluation, Model_Training, Prediction, Climate_Text_Analysis
def main():
    st.set_page_config(
        page_title = "Climate Temperature Prediction",
        page_icon = "🌡️",
        layout = 'wide'
    )

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

    # Dictionary to map page names to their corresponding modules and methods
    page_modules = {
        "Home": Home,
        "Exploratory Data Analysis": EDA,
        "Feature Engineering": Feature_Engineering,
        "Model Training": Model_Training,
        "Model Evaluation": Model_Evaluation,
        "Prediction": Prediction,
        "Climate Text Analysis": Climate_Text_Analysis
    }

    # Display the selected page
    try:
        selected_page = page_modules.get(page)
        if selected_page:
            selected_page.show()
        else:
            st.error("Page not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
if __name__ == "__main__":
    main()
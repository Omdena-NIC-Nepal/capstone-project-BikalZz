# Climate Temperature Prediction Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [FAQ](#faq)
7. [Troubleshooting](#troubleshooting)

## Project Overview

This project is a Streamlit-based web application for analyzing climate data and predicting maximum temperatures based on various environmental and agricultural factors. The application includes:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Training
- Model Evaluation
- Prediction Interface
- Natural Language Processing (NLP) capabilities

**Target Variable**: Average Maximum Temperature (`avg_max_temp`)

## Features

### Core Features
- **Interactive Data Exploration**: Visualize trends, distributions, and correlations
- **Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression, Ridge Regression
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Feature Engineering**: Create and select features for modeling
- **Prediction Interface**: Make new predictions with trained models
- **NLP Integration**: Analyze climate-related text using spaCy

### Technical Highlights
- Modular architecture with separate pages
- Session state management for data persistence
- Model saving functionality
- Responsive UI with interactive components

## Installation

### Prerequisites
- Python 3.8+

### Setup Instructions
1. Clone the repository:
   ```cmd
   git clone https://github.com/Omdena-NIC-Nepal/capstone-project-BikalZz.git
   cd capstone-project-BikalZz
   ```

2. Create and activate a virtual environment (recommended):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

4. Download the spaCy language model:
   ```cmd
   python -m spacy download en_core_web_sm
   ```

5. Place your `combined_data.csv` file in the directory: `../data/processed_data'

## Usage

### Running the Application
Start the Streamlit application:
```cmd
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Navigation Guide
1. **Home**: Project introduction and NLP demo
2. **EDA**: Explore and visualize the data
3. **Feature Engineering**: Create and select features
4. **Model Training**: Train machine learning models
5. **Model Evaluation**: Compare model performance
6. **Prediction**: Make new predictions
7. **Climate Text Analysis**: Analyze the climate articles for climate sentimental analysis.

## File Structure

```
capstone-project-BikalZz/
│
├── data/                # Main application file
│   ├── raw/       # raw datas
│   │   ├── climate       # raw climate datas
│   │   │   ├── npl-rainfall-adm2-full.csv       # rainfall data
│   │   │   ├── observed_annual-average-largest-1-day-precipitation.csv       # precipitation data
│   │   │   ├── observed-annual-average_temp.csv       # annual average mean temperature data
│   │   │   ├── observed-annual-average-max-temp.csv       # annual average maximum temperature data
│   │   │   ├── observed-annual-average-min-temp.csv       # annual average minimum temperature data
│   │   │   └── observed-annual-relative-humidity.csv       # annual relative humidity data
│   │   └── socio-economic       # raw socio-economic datas
│   │       └── eco-socio-env-health-edu-dev-energy_npl.csv       # socio-economic data
│   ├── processed_data/       # raw datas
│   │   └── combined_data.csv       # combination of all the raw data with selected features and data preprocessing
│   └── sentiment_data/       # sentimental data on climate change for NLP
│       └── positive.csv       # positive sentiment dataset
│       └── negative.csv       # negative sentiment dataset
├── app.py                # Main application file
├── data_utils.py         # Utility functions for data processing
├── pages/
│   ├── Home.py       # Project introduction
│   ├── EDA.py        # Exploratory Data Analysis
│   ├── Model_Training.py  # Model training page
│   ├── Model_Evaluation.py # Model evaluation
│   ├── Prediction.py      # Prediction interface
│   └── Feature_Engineering.py # Feature engineering
├── requirements.txt      # Dependencies
├── data_preprocessing.ipynb               # combining the data from different data sources and handeling the missing and duplicate data and saved to `../data/processed_data/combined_data.csv`
├── data_sources.txt        # Sources of data with url
├── README.txt        # Project instruction
└── Documentation.md        # Documentation of the project with installation, FAQ, Troubleshooting guide
```

## FAQ

### Q1: Where is the data sources for this porject data?
A: You can see the sources for the data used in this project provided in the `data_sources.txt` file in root directory.

### Q2: How do I use my own dataset?
A: Replace `combined_data.csv` in the `../data/processed_data/` with your dataset file. Ensure it has similar structure and column names, or modify the code in `data_utils.py` to handle your specific format.

### Q3: Why aren't my trained models persisting between sessions?
A: The application saves models to disk in the `models/` directory. If you're not seeing previously trained models:
1. Check that the `models` directory exists.
2. Verify you have write permissions.
3. Ensure you're not clearing the directory between sessions.

### Q4: How can I add more machine learning models?
A: To add new models:
1. Add the model class to the `model_options` dictionary in `Model_Training.py`
2. Add any model-specific hyperparameters in the parameters section
3. The rest of the workflow will automatically adapt to the new model

### Q5: The NLP feature isn't working - what should I check?
A: If the NLP isn't working:
1. Verify you've installed the spaCy model (`python -m spacy download en_core_web_sm`)
2. Ensure you're entering text in the input box on the Home page/Climate Text Analysis page.

### Q6: How can I deploy this application?
A: You can deploy this Streamlit app on Streamlit Sharing. 

### Q7: What is the url of the project deployment?
A: You can view the app of this project through this link: https://omdena-nic-nepal-capstone-project-bikalzz.streamlit.app/

## Troubleshooting

### Common Issues and Solutions

**Issue**: Missing data file error  
**Solution**: Ensure `combined_data.csv` is in the `../data/processed_data/` directory with the correct name.

**Issue**: ModuleNotFoundError  
**Solution**: Verify all dependencies are installed (`pip install -r requirements.txt`)

**Issue**: Model training takes too long  
**Solution**: 
- Reduce dataset size for testing
- Use simpler models (Linear Regression instead of Random Forest)
- Decrease the number of estimators or tree depth

**Issue**: Visualizations not displaying properly  
**Solution**: 
- Check for null values in your data
- Ensure matplotlib/seaborn are properly installed
- Restart the Streamlit application

## Support

For additional support, please open an issue on the project's GitHub repository or contact the project maintainer.

---
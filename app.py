import pandas as pd
import numpy as np
from numpy import trapz
from scipy import integrate
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.meta import Stacking
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pickle

def load_data():
    df = pd.read_csv(r'dataset_da_2017.csv', encoding='utf-8')
    columns_to_drop = ['Patient ID', 'Year of diagnosis', 'Survival months flag']
    df = df.drop(columns=columns_to_drop)
    df.replace("Unknown", np.nan, inplace=True)
    return df

# Load dataset
df = load_data()

# Preprocess data
X = df.drop(['Survival months', 'Vital status'], axis=1)
y_surv = df['Survival months']
y_vital = df['Vital status']
x_train, x_test, y_surv_train, y_surv_test, y_vital_train, y_vital_test = train_test_split(
    X, y_surv, y_vital, test_size=0.3, random_state=42, stratify=y_vital)

# Define mapping dictionaries
income_mapping = {'< $35,000': 0, '$35,000 - $39,999': 1, '$40,000 - $44,999': 2, '$45,000 - $49,999': 3, '$50,000 - $54,999': 4,
                  '$55,000 - $59,999': 5, '$60,000 - $64,999': 6, '$65,000 - $69,999': 7, '$70,000 - $74,999': 8, '≥ $75,000': 9}
grade_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
stage_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
summary_stage_mapping = {'Localized': 0, 'Regional': 1, 'Distant': 2}

# Function to load the preprocessed data
def load_preprocessed_data():
    with open(r'C:\Users\ming\Desktop\python_work\app\x_train_df.pkl', 'rb') as f:
        x_train_df = pickle.load(f)
    with open(r'C:\Users\ming\Desktop\python_work\app\x_test_df.pkl', 'rb') as f:
        x_test_df = pickle.load(f)
    return x_train_df, x_test_df

# Load preprocessed data
x_train_df, x_test_df = load_preprocessed_data()

# Reset indices to ensure consistency
x_train_df = x_train_df.reset_index(drop=True)
x_test_df = x_test_df.reset_index(drop=True)
y_surv_train = y_surv_train.reset_index(drop=True)
y_surv_test = y_surv_test.reset_index(drop=True)
y_vital_train = y_vital_train.reset_index(drop=True)
y_vital_test = y_vital_test.reset_index(drop=True)

# Combine features and target variables
train_data_new = pd.concat([x_train_df, y_surv_train, y_vital_train], axis=1)
test_data_new = pd.concat([x_test_df, y_surv_test, y_vital_test], axis=1)

# Map 'Alive' to 0 and 'Dead' to 1
train_data_new['Vital status'] = train_data_new['Vital status'].map({'Alive': 0, 'Dead': 1})
test_data_new['Vital status'] = test_data_new['Vital status'].map({'Alive': 0, 'Dead': 1})

# Function to convert DataFrame to structured array for scikit-survival
def df_to_structured(df, event_col, time_col):
    return np.array([(e, t) for e, t in zip(df[event_col], df[time_col])], dtype=[('event', bool), ('time', np.float64)])

# Convert training and testing data
y_train_structured = df_to_structured(train_data_new, 'Vital status', 'Survival months')
y_test_structured = df_to_structured(test_data_new, 'Vital status', 'Survival months')

# CSS for custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #e0e3e8;
        padding: 20px;
    }
    .sidebar .sidebar-content h2 {
        color: #004085;
    }
    .main .block-container {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .main .block-container h1, h2, h3, h4 {
        color: #004085;
    }
    .stButton button {
        background-color: #004085;
        color: white;
    }
    .stButton button:hover {
        background-color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.title('NSCLC Survival Prediction')

st.sidebar.header('Input Variables')
def user_input_features():
    variables = {
        "Sex": st.sidebar.selectbox("Sex", options=["Male", "Female"]),
        "Age": st.sidebar.number_input("Age", min_value=0),
        "Race": st.sidebar.selectbox("Race", options=["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"]),
        "Median household income": st.sidebar.selectbox("Median household income", options=["< $35,000", "$35,000 - $39,999", "$40,000 - $44,999", "$45,000 - $49,999", "$50,000 - $54,999",
                          "$55,000 - $59,999", "$60,000 - $64,999", "$65,000 - $69,999", "$70,000 - $74,999", "≥ $75,000"]),
        "Marital status at diagnosis": st.sidebar.selectbox("Marital status at diagnosis", options=["Single (never married)", "Married (including common law)", "Divorced", "Widowed", "Separated", "Unmarried or Domestic Partner"]),
        "Tumor Size": st.sidebar.number_input("Tumor size (mm)", min_value=0),
        "Grade": st.sidebar.selectbox("Grade", options=["I", "II", "III", "IV"]),
        "Stage Group": st.sidebar.selectbox("Stage Group", options=["I", "II", "III", "IV"]),
        "Combined Summary Stage": st.sidebar.selectbox("Combined Summary Stage", options=["Localized", "Regional", "Distant"]),
        "Surgery": st.sidebar.selectbox("Surgery", options=["Yes", "No"]),
        "Radiation": st.sidebar.selectbox("Radiation", options=["Yes", "No"]),
        "Chemotherapy": st.sidebar.selectbox("Chemotherapy", options=["Yes", "No"]),
        "Mets at DX-bone": st.sidebar.selectbox("Mets at DX-bone", options=["Yes", "No"]),
        "Mets at DX-brain": st.sidebar.selectbox("Mets at DX-brain", options=["Yes", "No"]),
        "Mets at DX-lung": st.sidebar.selectbox("Mets at DX-lung", options=["Yes", "No"]),
        "Mets at DX-liver": st.sidebar.selectbox("Mets at DX-liver", options=["Yes", "No"])
    }
    
    # Convert user inputs to DataFrame
    input_data = {
        "Sex_Male": 1 if variables["Sex"] == "Male" else 0,
        "Sex_Female": 1 if variables["Sex"] == "Female" else 0,
        "Age": variables["Age"],
        "Race_White": 1 if variables["Race"] == "White" else 0,
        "Race_Black": 1 if variables["Race"] == "Black" else 0,
        "Race_Asian or Pacific Islander": 1 if variables["Race"] == "Asian or Pacific Islander" else 0,
        "Race_American Indian/Alaska Native": 1 if variables["Race"] == "American Indian/Alaska Native" else 0,
        "Median household income": income_mapping[variables["Median household income"]],
        "Marital status at diagnosis_Single (never married)": 1 if variables["Marital status at diagnosis"] == "Single (never married)" else 0,
        "Marital status at diagnosis_Married (including common law)": 1 if variables["Marital status at diagnosis"] == "Married (including common law)" else 0,
        "Marital status at diagnosis_Divorced": 1 if variables["Marital status at diagnosis"] == "Divorced" else 0,
        "Marital status at diagnosis_Widowed": 1 if variables["Marital status at diagnosis"] == "Widowed" else 0,
        "Marital status at diagnosis_Separated": 1 if variables["Marital status at diagnosis"] == "Separated" else 0,
        "Marital status at diagnosis_Unmarried or Domestic Partner": 1 if variables["Marital status at diagnosis"] == "Unmarried or Domestic Partner" else 0,
        "Tumor Size": variables["Tumor Size"],
        "Grade": grade_mapping[variables["Grade"]],
        "Stage Group": stage_mapping[variables["Stage Group"]],
        "Combined Summary Stage": summary_stage_mapping[variables["Combined Summary Stage"]],
        "Surgery_Yes": 1 if variables["Surgery"] == "Yes" else 0,
        "Surgery_No": 1 if variables["Surgery"] == "No" else 0,
        "Radiation_Yes": 1 if variables["Radiation"] == "Yes" else 0,
        "Radiation_No": 1 if variables["Radiation"] == "No" else 0,
        "Chemotherapy_Yes": 1 if variables["Chemotherapy"] == "Yes" else 0,
        "Chemotherapy_No": 1 if variables["Chemotherapy"] == "No" else 0,
        "Mets at DX-bone_Yes": 1 if variables["Mets at DX-bone"] == "Yes" else 0,
        "Mets at DX-bone_No": 1 if variables["Mets at DX-bone"] == "No" else 0,
        "Mets at DX-brain_Yes": 1 if variables["Mets at DX-brain"] == "Yes" else 0,
        "Mets at DX-brain_No": 1 if variables["Mets at DX-brain"] == "No" else 0,
        "Mets at DX-lung_Yes": 1 if variables["Mets at DX-lung"] == "Yes" else 0,
        "Mets at DX-lung_No": 1 if variables["Mets at DX-lung"] == "No" else 0,
        "Mets at DX-liver_Yes": 1 if variables["Mets at DX-liver"] == "Yes" else 0,
        "Mets at DX-liver_No": 1 if variables["Mets at DX-liver"] == "No" else 0
    }
    
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Function to preprocess user input
def preprocess_user_input(input_df):
    # Ensure all columns in input_df
    for col in x_train_df.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Order columns as in training data
    input_df = input_df[x_train_df.columns]

    # Convert certain columns to integer
    cols_to_convert = ['Age', 'Median household income', 'Grade', 'Stage Group', 'Combined Summary Stage']
    for col in cols_to_convert:
        input_df[col] = input_df[col].astype(int)

    # Normalize using StandardScaler
    scaler = StandardScaler()
    columns_to_scale = ['Age', 'Tumor Size']
    input_df[columns_to_scale] = scaler.fit_transform(input_df[columns_to_scale])

    return input_df

input_df = preprocess_user_input(input_df)

# Add model selection
model_choice = st.sidebar.selectbox('Choose Model', options=['Cox Proportional Hazards', 'Survival Tree', "RandomSurvivalForest", "Gradient Boosting Survival", "Stacking" ])

# Add a prediction button
predict_button = st.sidebar.button('Predict')

# Load the models
with open(r'C:\Users\ming\Desktop\python_work\app\coxph_model.pkl', 'rb') as f:
    coxph = pickle.load(f)

with open(r'C:\Users\ming\Desktop\python_work\app\rsf_model.pkl', 'rb') as f:
    rsf = pickle.load(f)

with open(r'C:\Users\ming\Desktop\python_work\app\gbm_model.pkl', 'rb') as f:
    gbm = pickle.load(f)

with open(r'C:\Users\ming\Desktop\python_work\app\tree_model.pkl', 'rb') as f:
    tree = pickle.load(f)

with open(r'C:\Users\ming\Desktop\python_work\app\stacking_model.pkl', 'rb') as f:
    stacking_model = pickle.load(f)

# Use the loaded models to predict
if predict_button:
    if model_choice == 'Cox Proportional Hazards':
        # Predict mean survival time
        predicted_mean_survival_time = coxph.predict_expectation(input_df).iloc[0]
        st.subheader(f'Predicted Mean Survival Time (Cox): {predicted_mean_survival_time:.2f} months')
        
        # Estimate survival function
        survival_function = coxph.predict_survival_function(input_df, times=np.linspace(0, max(y_surv_train), 100))
        
        # Visualize the survival curve
        st.subheader('Survival Curve')
        fig, ax = plt.subplots()
        ax.step(survival_function.index, survival_function.iloc[:, 0], where='post')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curve (Cox)')
        st.pyplot(fig)
        
    elif model_choice == 'Survival Tree':
        # Predict survival function using survival tree
        survival_function_tree = tree.predict_survival_function(input_df)
        # Function to calculate mean survival time
        def mean_survival_time(surv_funcs):
            mean_times = []
            for surv_func in surv_funcs:
                times = surv_func.x
                survival_probs = surv_func.y
                mean_time = np.trapz(survival_probs, times)
                mean_times.append(mean_time)
            return np.array(mean_times)
        
        # Calculate mean survival time 
        predicted_means_tree = mean_survival_time(survival_function_tree)
        st.subheader(f'Predicted Mean Survival Time (ST): {predicted_means_tree[0]:.2f} months')
        
        # Visualize the survival curve
        st.subheader('Survival Curve')
        fig, ax = plt.subplots()
        for surv_func in survival_function_tree:
            ax.step(surv_func.x, surv_func.y, where='post')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curve (ST)')
        st.pyplot(fig)

    elif model_choice == 'RandomSurvivalForest':
        # Predict survival function using rsf
        survival_function_rsf = rsf.predict_survival_function(input_df)
        # Function to calculate mean survival time
        def mean_survival_time(surv_funcs):
            mean_times = []
            for surv_func in surv_funcs:
                times = surv_func.x
                survival_probs = surv_func.y
                mean_time = np.trapz(survival_probs, times)
                mean_times.append(mean_time)
            return np.array(mean_times)

        # Calculate mean survival time for rsf
        predicted_means_rsf = mean_survival_time(survival_function_rsf)
        st.subheader(f'Predicted Mean Survival Time (RSF): {predicted_means_rsf[0]:.2f} months')
        
        # Visualize the survival curve
        st.subheader('Survival Curve')
        fig, ax = plt.subplots()
        for surv_func in survival_function_rsf:
            ax.step(surv_func.x, surv_func.y, where='post')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curve (RSF)')
        st.pyplot(fig)
    
    elif model_choice == 'Gradient Boosting Survival':
        # Predict survival function using gbsa
        survival_function_gbm = gbm.predict_survival_function(input_df)
        # Function to calculate mean survival time
        def mean_survival_time(surv_funcs):
            mean_times = []
            for surv_func in surv_funcs:
                times = surv_func.x
                survival_probs = surv_func.y
                mean_time = np.trapz(survival_probs, times)
                mean_times.append(mean_time)
            return np.array(mean_times)

        # Calculate mean survival time for gbsa
        predicted_means_gbm = mean_survival_time(survival_function_gbm)
        st.subheader(f'Predicted Mean Survival Time (GBSA): {predicted_means_gbm[0]:.2f} months')
        
        # Visualize the survival curve
        st.subheader('Survival Curve')
        fig, ax = plt.subplots()
        for surv_func in survival_function_gbm:
            ax.step(surv_func.x, surv_func.y, where='post')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curve (GBSA)')
        st.pyplot(fig)   
        
    elif model_choice == 'Stacking':
        # Predict survival function using stacking
        survival_function_sta = stacking_model.predict_survival_function(input_df)
        # Function to calculate mean survival time
        def mean_survival_time(surv_funcs):
            mean_times = []
            for surv_func in surv_funcs:
                times = surv_func.x
                survival_probs = surv_func.y
                mean_time = np.trapz(survival_probs, times)
                mean_times.append(mean_time)
            return np.array(mean_times)

        # Calculate mean survival time for stacking
        predicted_means_sta = mean_survival_time(survival_function_sta)
        st.subheader(f'Predicted Mean Survival Time (Stacking): {predicted_means_sta[0]:.2f} months')
        
        # Visualize the survival curve
        st.subheader('Survival Curve')
        fig, ax = plt.subplots()
        for surv_func in survival_function_sta:
            ax.step(surv_func.x, surv_func.y, where='post')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curve (Stacking)')
        st.pyplot(fig)

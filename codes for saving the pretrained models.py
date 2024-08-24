import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.meta import Stacking
from sksurv.linear_model import CoxPHSurvivalAnalysis

def load_data():
    df = pd.read_csv('dataset_da_2017.csv', encoding='utf-8')
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

def preprocess_data(x_train, x_test):
    # Encode the ordinal variables
    x_train['Median household income'] = x_train['Median household income'].map(income_mapping)
    x_test['Median household income'] = x_test['Median household income'].map(income_mapping)

    x_train['Grade'] = x_train['Grade'].map(grade_mapping)
    x_test['Grade'] = x_test['Grade'].map(grade_mapping)

    x_train['Stage Group'] = x_train['Stage Group'].map(stage_mapping)
    x_test['Stage Group'] = x_test['Stage Group'].map(stage_mapping)

    x_train['Combined Summary Stage'] = x_train['Combined Summary Stage'].map(summary_stage_mapping)
    x_test['Combined Summary Stage'] = x_test['Combined Summary Stage'].map(summary_stage_mapping)

    # One-Hot encoding for nominal variables
    nominal_vars = ["Sex", "Race", "Marital status at diagnosis", "Surgery", "Radiation",
                    "Chemotherapy", "Mets at DX-bone", "Mets at DX-brain", "Mets at DX-lung",
                    "Mets at DX-liver"]

    x_train = pd.get_dummies(x_train, columns=nominal_vars)
    x_test = pd.get_dummies(x_test, columns=nominal_vars)

    x_train["Tumor Size"] = pd.to_numeric(x_train["Tumor Size"], errors="coerce")
    x_test["Tumor Size"] = pd.to_numeric(x_test["Tumor Size"], errors="coerce")

    # Save column names and indices before imputation
    columns = x_train.columns.tolist()
    train_index = x_train.index
    test_index = x_test.index

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

    # Convert back to pandas dataframe
    x_train_df = pd.DataFrame(x_train_imputed, columns=columns, index=train_index)
    x_test_df = pd.DataFrame(x_test_imputed, columns=columns, index=test_index)

    # Convert certain columns to integer
    cols_to_convert = ['Age', 'Median household income', 'Grade', 'Stage Group', 'Combined Summary Stage']
    for col in cols_to_convert:
        x_train_df[col] = x_train_df[col].astype(int)
        x_test_df[col] = x_test_df[col].astype(int)

    # Normalize using StandardScaler
    scaler = StandardScaler()
    columns_to_scale = ['Age', 'Tumor Size']
    x_train_df[columns_to_scale] = scaler.fit_transform(x_train_df[columns_to_scale])
    x_test_df[columns_to_scale] = scaler.transform(x_test_df[columns_to_scale])

    return x_train_df, x_test_df

# Preprocess train and test data
x_train_df, x_test_df = preprocess_data(x_train, x_test)

# Save preprocessed data
with open('x_train_df.pkl', 'wb') as f:
    pickle.dump(x_train_df, f)

with open('x_test_df.pkl', 'wb') as f:
    pickle.dump(x_test_df, f)

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

# Initialize and train the models
coxph = CoxPHFitter(penalizer=0.3593813663804626)
coxph.fit(train_data_new, duration_col='Survival months', event_col='Vital status')

rsf = RandomSurvivalForest(min_samples_leaf=5, min_samples_split=10, n_estimators=50, n_jobs=-1, random_state=42)
rsf.fit(x_train_df, y_train_structured)

gbm = GradientBoostingSurvivalAnalysis(random_state=42)
gbm.fit(x_train_df, y_train_structured)

tree = SurvivalTree(max_depth=5, max_features=0.1, min_samples_leaf=5, min_samples_split=5, random_state=42)
tree.fit(x_train_df, y_train_structured)

meta_model = CoxPHSurvivalAnalysis(alpha=1250)
stacking_model = Stacking(
    meta_estimator=meta_model,
    base_estimators=[
        ("rsf", rsf),
        ("gbsa", gbm),
        ("tree", tree)
    ],
    probabilities=False
)
stacking_model.fit(x_train_df, y_train_structured)

# Save the models
with open('coxph_model.pkl', 'wb') as f:
    pickle.dump(coxph, f)

with open('rsf_model.pkl', 'wb') as f:
    pickle.dump(rsf, f)

with open('gbm_model.pkl', 'wb') as f:
    pickle.dump(gbm, f)

with open('tree_model.pkl', 'wb') as f:
    pickle.dump(tree, f)

with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)
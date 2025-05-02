from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


import pandas as pd
import pickle
import os

# Get the directory of the current script (somehow you need this on vscode but not on jupyter lab)
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Load and preprocess data
df = pd.read_csv('APS.csv')
df.drop(columns=['id', 'Arrival Delay in Minutes'], axis=1, inplace=True)

# Encode categorical columns
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Encode target column
le = LabelEncoder()
df_encoded['satisfaction'] = le.fit_transform(df_encoded['satisfaction'])

X = df_encoded.drop(columns='satisfaction')
y = df_encoded['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create base models
knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

rf = RandomForestClassifier()
xgb = XGBClassifier(eval_metric='logloss')

# Voting Classifier (Soft voting works best if models have `predict_proba`)
voting_clf = VotingClassifier(
    estimators=[('knn', knn), ('rf', rf), ('xgb', xgb)],
    voting='soft'  # or 'hard'
)

# Train the ensemble
voting_clf.fit(X_train, y_train)

# Save the ensemble model
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
    
# Save column names from X_train after encoding
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("✅ Ensemble model saved as 'ensemble_model.pkl'")

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. SETUP PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "*.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_clean_data(file_pattern):
    print("Reading CSV files... this may take a minute on a Mac Pro.")
    all_files = glob.glob(file_pattern)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found matching: {file_pattern}")
    # Reading files into a list and concatenating
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Cleaning Step A: Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Cleaning Step B: Drop columns with 0 variance or redundant info
    # 'Fwd Header Length.1' is often a duplicate in the Kaggle version
    if 'Fwd Header Length.1' in df.columns:
        df = df.drop(columns=['Fwd Header Length.1'])
        
    # Cleaning Step C: Handle Infs and NaNs
    print("Handling Infinite and Null values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Cleaning Step D: Remove duplicates (crucial for research integrity)
    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {before_dup - len(df)} duplicates.")
    
    return df

# 2. EXECUTION
df = load_and_clean_data(DATA_PATH)

# Encoding the labels (Benign, DoS, PortScan, etc.)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

# 3. FEATURE SELECTION & SPLIT
X = df.drop(columns=['Label'])
y = df['Label']

# Memory efficiency: Downcast floats to save RAM on the Mac Pro
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling (Models perform better when features are on the same scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# 4. TRAINING BASE DETECTOR 1: RANDOM FOREST
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42) # n_jobs=-1 uses all Mac Pro cores
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"RF Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")

# 5. TRAINING BASE DETECTOR 2: XGBOOST
print("\n--- Training XGBoost ---")
# Use hist for faster training on large datasets
xgb = XGBClassifier(tree_method='hist', n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"XGB Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
joblib.dump(xgb, f"{MODEL_DIR}/xgboost_model.pkl")

print("\nAll models saved in the /models directory. Step 1 Complete.")
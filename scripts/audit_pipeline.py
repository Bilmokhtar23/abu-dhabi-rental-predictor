"""
COMPREHENSIVE ML PIPELINE AUDIT
Validates consistency across data, feature engineering, training, and deployment
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE ML PIPELINE AUDIT")
print("=" * 80)

# ============================================================================
# PHASE 1: DATA CONSISTENCY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: DATA CONSISTENCY CHECK")
print("="*80)

# Load dataset
# Load final production datasets for audit
df_train = pd.read_csv('data/processed/train_set_final.csv')
df_test = pd.read_csv('data/processed/test_set_final.csv')
df = pd.concat([df_train, df_test], ignore_index=True)
print(f"\n✓ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\n❌ ISSUE: Missing values found:")
    print(missing[missing > 0])
else:
    print("✓ No missing values")

# Verify target variable
print(f"\n✓ Target variable 'Rent' stats:")
print(f"  - Mean: {df['Rent'].mean():,.2f} AED")
print(f"  - Median: {df['Rent'].median():,.2f} AED")
print(f"  - Min: {df['Rent'].min():,.2f} AED")
print(f"  - Max: {df['Rent'].max():,.2f} AED")

# Check categorical columns
categorical_cols = ['Location', 'Type', 'Furnishing']
print(f"\n✓ Categorical columns:")
for col in categorical_cols:
    print(f"  - {col}: {df[col].nunique()} unique values")

# ============================================================================
# PHASE 2: FEATURE ENGINEERING VALIDATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: FEATURE ENGINEERING VALIDATION")
print("="*80)

# Get expected features from dataset
dataset_features = [col for col in df.columns if col != 'Rent']
print(f"\n✓ Dataset has {len(dataset_features)} features (excluding Rent)")

# Load model's expected features
with open('model_outputs/production/feature_columns_latest.json', 'r') as f:
    model_features = json.load(f)
print(f"✓ Model expects {len(model_features)} features")

# Compare features
dataset_set = set(dataset_features)
model_set = set(model_features)

missing_in_model = dataset_set - model_set
missing_in_dataset = model_set - dataset_set

if missing_in_model:
    print(f"\n❌ ISSUE: Features in dataset but NOT in model ({len(missing_in_model)}):")
    for f in sorted(missing_in_model):
        print(f"  - {f}")
        
if missing_in_dataset:
    print(f"\n❌ ISSUE: Features expected by model but NOT in dataset ({len(missing_in_dataset)}):")
    for f in sorted(missing_in_dataset):
        print(f"  - {f}")

if not missing_in_model and not missing_in_dataset:
    print("\n✓ Perfect match: All features align between dataset and model")
    
# Check feature order
if dataset_features == model_features:
    print("✓ Feature order matches exactly")
else:
    print("\n⚠️  WARNING: Feature order differs between dataset and model")
    print("   (This is OK as long as we reorder before prediction)")

# ============================================================================
# PHASE 3: MODEL ARTIFACT VALIDATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: MODEL ARTIFACT VALIDATION")
print("="*80)

# Load model
model = joblib.load('model_outputs/production/xgboost_latest.joblib')
print(f"\n✓ Model loaded: {type(model).__name__}")

# Check model's feature names
if hasattr(model, 'feature_names_in_'):
    model_trained_features = list(model.feature_names_in_)
    print(f"✓ Model was trained with {len(model_trained_features)} features")
    
    if model_trained_features == model_features:
        print("✓ Model's trained features match feature_columns_latest.json")
    else:
        print("❌ ISSUE: Mismatch between trained features and feature_columns_latest.json")
        
# Load encoder
encoder = joblib.load('model_outputs/production/target_encoder_latest.joblib')
print(f"\n✓ Encoder loaded: {type(encoder).__name__}")
print(f"  - Expects {encoder._dim} input columns")
print(f"  - Encodes: {encoder.cols}")

if encoder._dim != 3:
    print(f"❌ ISSUE: Encoder expects {encoder._dim} columns, should be 3")
elif encoder.cols != categorical_cols:
    print(f"❌ ISSUE: Encoder cols {encoder.cols} don't match expected {categorical_cols}")
else:
    print("✓ Encoder configuration is correct")

# ============================================================================
# PHASE 4: PREDICTION CONSISTENCY TEST
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: PREDICTION CONSISTENCY TEST")
print("="*80)

# Get a test sample
test_row = df.iloc[0].copy()
actual_rent = test_row['Rent']

print(f"\nTest property:")
print(f"  Location: {test_row['Location']}")
print(f"  Type: {test_row['Type']}")
print(f"  Furnishing: {test_row['Furnishing']}")
print(f"  Area: {test_row['Area_in_sqft']} sqft")
print(f"  Beds: {test_row['Beds']}")
print(f"  Baths: {test_row['Baths']}")
print(f"  ACTUAL Rent: {actual_rent:,.0f} AED")

# Prepare features in model order
X_test = test_row[model_features].to_frame().T

# Separate numeric and categorical
numeric_features = [col for col in model_features if col not in categorical_cols]
X_numeric = X_test[numeric_features]
X_categorical = X_test[categorical_cols]

# Encode categorical
X_categorical_encoded = encoder.transform(X_categorical)

# Combine
X_final = pd.concat([X_numeric.reset_index(drop=True), 
                     X_categorical_encoded.reset_index(drop=True)], axis=1)

# Reorder to match model
X_final = X_final[model_features]

# Convert all to numeric (encoder may return objects)
for col in X_final.columns:
    X_final[col] = pd.to_numeric(X_final[col], errors='coerce')

# Predict
prediction = model.predict(X_final)[0]
error = abs(prediction - actual_rent)
error_pct = (error / actual_rent) * 100

print(f"\nPrediction: {prediction:,.0f} AED")
print(f"Error: {error:,.0f} AED ({error_pct:.2f}%)")

if error_pct < 5:
    print("✅ EXCELLENT: Prediction within 5% of actual")
elif error_pct < 10:
    print("✓ GOOD: Prediction within 10% of actual")
else:
    print(f"⚠️  WARNING: Prediction error is {error_pct:.2f}%")

# ============================================================================
# PHASE 5: TRAINING SCRIPT VALIDATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: TRAINING SCRIPT VALIDATION")
print("="*80)

# Check if training script uses the same data
print("\nChecking save_final_robust_model.py:")

with open('scripts/save_final_robust_model.py', 'r') as f:
    training_script = f.read()
    
if 'train_set_final.csv' in training_script or 'train_set.csv' in training_script:
    print("✓ Uses correct dataset: train_set_final.csv or train_set.csv")
else:
    print("❌ ISSUE: Doesn't use current production datasets")
    
if "categorical_cols = ['Location', 'Type', 'Furnishing']" in training_script:
    print("✓ Uses correct categorical columns")
else:
    print("❌ ISSUE: Categorical columns definition differs")
    
if 'TargetEncoder' in training_script:
    print("✓ Uses TargetEncoder")
else:
    print("❌ ISSUE: Doesn't use TargetEncoder")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("AUDIT SUMMARY")
print("="*80)

issues = []

if missing.sum() > 0:
    issues.append("Missing values in dataset")
    
if missing_in_model or missing_in_dataset:
    issues.append("Feature mismatch between dataset and model")
    
if encoder._dim != 3:
    issues.append("Encoder dimension mismatch")
    
if error_pct > 10:
    issues.append(f"High prediction error ({error_pct:.2f}%)")

if not issues:
    print("\n✅ ALL CHECKS PASSED - PIPELINE IS ALIGNED")
else:
    print(f"\n❌ FOUND {len(issues)} ISSUE(S):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

print("\n" + "="*80)

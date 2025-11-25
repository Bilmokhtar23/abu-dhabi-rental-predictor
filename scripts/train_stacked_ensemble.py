"""
Phase 4: Ensemble Stacking
Combines predictions from XGBoost, LightGBM, and CatBoost using a meta-learner.
Expected improvement: -150 to -300 AED MAE
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Configuration
TRAIN_PATH = Path("data/processed/train_set_FINAL.csv")
VAL_PATH = Path("data/processed/val_set_FINAL.csv")
TEST_PATH = Path("data/processed/test_set_FINAL.csv")
BEST_PARAMS_PATH = Path("model_outputs/tuned/best_params_20251123_185430.json")
RANDOM_STATE = 42
N_FOLDS = 5  # For generating meta-features


def load_data():
    """Load pre-split FINAL datasets (Issue #2 & #3 fix)"""
    
    print("="*100)
    print("DATA LOADING")
    print("="*100)
    
    # Load pre-split datasets
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"\n✅ Loaded pre-split datasets:")
    print(f"   Train: {len(train_df):,} properties × {len(train_df.columns)} columns")
    print(f"   Val:   {len(val_df):,} properties × {len(val_df.columns)} columns")
    print(f"   Test:  {len(test_df):,} properties × {len(test_df.columns)} columns")
    
    # Note: "Duplicates" in these files are actually different properties
    # with identical feature values - this is valid market data, keep all rows
    
    # Split features and target
    X_train = train_df.drop('Rent', axis=1)
    y_train = train_df['Rent']
    
    X_val = val_df.drop('Rent', axis=1)
    y_val = val_df['Rent']
    
    X_test = test_df.drop('Rent', axis=1)
    y_test = test_df['Rent']
    
    feature_cols = X_train.columns.tolist()
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def load_best_params():
    """Load best hyperparameters from Phase 2"""
    
    print(f"\n🔧 Loading best hyperparameters from Phase 2...")
    
    with open(BEST_PARAMS_PATH, 'r') as f:
        all_params = json.load(f)
    
    print(f"   ✅ Loaded parameters for XGBoost, LightGBM, CatBoost")
    
    return all_params


def prepare_features(X_train, X_val, X_test, y_train, feature_cols):
    """Encode categorical features with OneHotEncoder"""

    categorical_cols = ['Location', 'Type', 'Furnishing']

    # Use OneHotEncoder for categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()

    # Fit encoder on training data
    encoded_train = encoder.fit_transform(X_train[categorical_cols])
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=X_train.index)

    # Transform validation and test data
    encoded_val = encoder.transform(X_val[categorical_cols])
    encoded_val_df = pd.DataFrame(encoded_val, columns=encoder.get_feature_names_out(categorical_cols), index=X_val.index)

    encoded_test = encoder.transform(X_test[categorical_cols])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols), index=X_test.index)

    # Drop original categorical columns and add encoded ones
    X_train_encoded = X_train_encoded.drop(categorical_cols, axis=1)
    X_val_encoded = X_val_encoded.drop(categorical_cols, axis=1)
    X_test_encoded = X_test_encoded.drop(categorical_cols, axis=1)

    X_train_encoded = pd.concat([X_train_encoded, encoded_train_df], axis=1)
    X_val_encoded = pd.concat([X_val_encoded, encoded_val_df], axis=1)
    X_test_encoded = pd.concat([X_test_encoded, encoded_test_df], axis=1)
    
    return X_train_encoded, X_val_encoded, X_test_encoded, encoder


def generate_meta_features(X_train, y_train, base_models, n_folds=5):
    """
    Generate meta-features using K-fold cross-validation.
    This prevents data leakage by ensuring predictions are made on out-of-fold data.
    """
    
    print(f"\n🔧 Generating meta-features using {n_folds}-fold CV...")
    
    # Initialize meta-feature arrays
    meta_features = np.zeros((X_train.shape[0], len(base_models)))
    
    # K-Fold split
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # For each base model
    for model_idx, (model_name, model) in enumerate(base_models.items()):
        print(f"   Processing {model_name}...")
        
        # For each fold
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            # Split data
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            # Train model on fold
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_fold_train, y_fold_train)
            
            # Predict on validation fold
            meta_features[val_idx, model_idx] = model_fold.predict(X_fold_val)
    
    print(f"   ✅ Generated meta-features shape: {meta_features.shape}")
    
    return meta_features


def train_base_models(X_train, y_train, params_dict):
    """Train base models with tuned hyperparameters"""
    
    print("\n" + "="*100)
    print("TRAINING BASE MODELS")
    print("="*100)
    
    base_models = {}
    
    # 1. XGBoost
    print(f"\n🏋️  Training XGBoost...")
    xgb_model = XGBRegressor(**params_dict['XGBoost'], random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train)
    base_models['XGBoost'] = xgb_model
    print(f"   ✅ XGBoost trained")
    
    # 2. LightGBM
    print(f"\n🏋️  Training LightGBM...")
    lgb_model = LGBMRegressor(**params_dict['LightGBM'], random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    base_models['LightGBM'] = lgb_model
    print(f"   ✅ LightGBM trained")
    
    # 3. CatBoost
    print(f"\n🏋️  Training CatBoost...")
    cat_model = CatBoostRegressor(**params_dict['CatBoost'], random_state=RANDOM_STATE, verbose=0)
    cat_model.fit(X_train, y_train)
    base_models['CatBoost'] = cat_model
    print(f"   ✅ CatBoost trained")
    
    return base_models


def train_meta_model(meta_features_train, y_train, meta_features_test=None):
    """Train meta-learner (Ridge regression) on base model predictions"""
    
    print("\n" + "="*100)
    print("TRAINING META-LEARNER")
    print("="*100)
    
    print(f"\n🏋️  Training Ridge meta-model...")
    print(f"   Meta-features shape: {meta_features_train.shape}")
    
    # Try different alpha values
    best_alpha = None
    best_score = float('inf')
    
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        meta_model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(meta_model, meta_features_train, y_train,
                                    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        
        if cv_mae < best_score:
            best_score = cv_mae
            best_alpha = alpha
    
    print(f"   Best alpha: {best_alpha} (CV MAE: {best_score:,.0f} AED)")
    
    # Train final meta-model
    meta_model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    meta_model.fit(meta_features_train, y_train)
    
    print(f"   ✅ Meta-model trained")
    
    # Print meta-model coefficients
    print(f"\n📊 Meta-model weights:")
    print(f"   XGBoost:  {meta_model.coef_[0]:.4f}")
    print(f"   LightGBM: {meta_model.coef_[1]:.4f}")
    print(f"   CatBoost: {meta_model.coef_[2]:.4f}")
    print(f"   Intercept: {meta_model.intercept_:.2f}")
    
    return meta_model, best_alpha


def evaluate_models(base_models, meta_model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate base models and stacked ensemble on validation and test sets"""
    
    print("\n" + "="*100)
    print("MODEL EVALUATION")
    print("="*100)
    
    results = []
    
    # Evaluate base models
    print(f"\n📊 Base Models Performance:")
    for model_name, model in base_models.items():
        # Validation set
        y_pred_val = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        # Test set
        y_pred_test = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results.append({
            'Model': model_name,
            'Type': 'Base',
            'Val MAE': val_mae,
            'Val R²': val_r2,
            'Test MAE': test_mae,
            'Test R²': test_r2
        })
        
        print(f"   {model_name:12s}  Val: {val_mae:,.0f} AED (R²={val_r2:.4f})  |  Test: {test_mae:,.0f} AED (R²={test_r2:.4f})")
    
    # Evaluate stacked ensemble
    print(f"\n📊 Stacked Ensemble Performance:")
    
    # Generate meta-features for train, val and test
    meta_features_train = np.column_stack([
        model.predict(X_train) for model in base_models.values()
    ])
    
    meta_features_val = np.column_stack([
        model.predict(X_val) for model in base_models.values()
    ])
    
    meta_features_test = np.column_stack([
        model.predict(X_test) for model in base_models.values()
    ])
    
    # Validation set
    y_pred_val = meta_model.predict(meta_features_val)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)
    
    # Test set
    y_pred_test = meta_model.predict(meta_features_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results.append({
        'Model': 'Stacked Ensemble',
        'Type': 'Ensemble',
        'Val MAE': val_mae,
        'Val R²': val_r2,
        'Test MAE': test_mae,
        'Test R²': test_r2
    })
    
    print(f"   {'Ensemble':12s}  Val: {val_mae:,.0f} AED (R²={val_r2:.4f})  |  Test: {test_mae:,.0f} AED (R²={test_r2:.4f})")
    
    return pd.DataFrame(results), val_mae, test_mae, test_r2, test_rmse


def main():
    """Main ensemble stacking pipeline"""
    
    print("="*100)
    print("PHASE 4: ENSEMBLE STACKING (UPDATED)")
    print("="*100)
    print(f"\nCombining XGBoost, LightGBM, and CatBoost with Ridge meta-learner")
    print(f"Using pre-split FINAL datasets (Issue #2 & #3 fixed)")
    
    # Load pre-split data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()
    
    # Load best hyperparameters
    params_dict = load_best_params()
    
    # Encode features
    print(f"\n🔧 Encoding categorical features...")
    X_train_enc, X_val_enc, X_test_enc, encoder = prepare_features(X_train, X_val, X_test, y_train, feature_cols)
    
    # Train base models
    base_models = train_base_models(X_train_enc, y_train, params_dict)
    
    # Generate meta-features (out-of-fold predictions)
    meta_features_train = generate_meta_features(X_train_enc, y_train, base_models, N_FOLDS)
    
    # Generate meta-features for test set
    print(f"\n🔧 Generating test meta-features...")
    meta_features_test = np.column_stack([
        model.predict(X_test_enc) for model in base_models.values()
    ])
    print(f"   ✅ Test meta-features shape: {meta_features_test.shape}")
    
    # Train meta-model
    meta_model, best_alpha = train_meta_model(meta_features_train, y_train, meta_features_test)
    
    # Evaluate all models
    results_df, val_mae, test_mae, test_r2, test_rmse = evaluate_models(
        base_models, meta_model, X_train_enc, X_val_enc, X_test_enc, y_train, y_val, y_test
    )
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*100)
    print("ENSEMBLE STACKING RESULTS")
    print("="*100)
    
    print("\n" + results_df.to_string(index=False))
    
    # Find best base model
    base_results = results_df[results_df['Type'] == 'Base']
    best_base = base_results.iloc[base_results['Test MAE'].argmin()]
    
    # Calculate improvement
    improvement = best_base['Test MAE'] - test_mae
    improvement_pct = (improvement / best_base['Test MAE']) * 100
    
    print(f"\n📊 Improvement over Best Base Model:")
    print(f"   Best Base ({best_base['Model']}): {best_base['Test MAE']:,.0f} AED")
    print(f"   Stacked Ensemble:  {test_mae:,.0f} AED")
    print(f"   Improvement: {improvement:+,.0f} AED ({improvement_pct:+.2f}%)")
    
    # Save results
    output_dir = Path("model_outputs/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comparison
    comparison_path = output_dir / f"ensemble_comparison_{timestamp}.csv"
    results_df.to_csv(comparison_path, index=False)
    
    # Save models
    ensemble_path = output_dir / f"stacked_ensemble_{timestamp}.joblib"
    joblib.dump({
        'base_models': base_models,
        'meta_model': meta_model,
        'encoder': encoder,
        'meta_alpha': best_alpha,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }, ensemble_path)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'base_models': list(base_models.keys()),
        'meta_model': 'Ridge',
        'meta_alpha': best_alpha,
        'n_folds_meta': N_FOLDS,
        'performance': {
            'val_mae': float(val_mae),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse)
        },
        'improvement_over_best_base': {
            'best_base_model': best_base['Model'],
            'best_base_mae': float(best_base['Test MAE']),
            'ensemble_mae': float(test_mae),
            'improvement_aed': float(improvement),
            'improvement_pct': float(improvement_pct)
        }
    }
    
    metadata_path = output_dir / f"ensemble_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   {comparison_path}")
    print(f"   {ensemble_path}")
    print(f"   {metadata_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*100)
    print("✅ PHASE 4 COMPLETE!")
    print("="*100)
    
    print(f"\n🏆 FINAL STACKED ENSEMBLE:")
    print(f"   Val MAE:   {val_mae:,.0f} AED")
    print(f"   Test MAE:  {test_mae:,.0f} AED")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:,.0f} AED")
    
    if improvement > 0:
        print(f"\n🎉 Ensemble IMPROVED over best base model:")
        print(f"   Improvement: {improvement:.0f} AED ({improvement_pct:.2f}%)")
    elif improvement < 0:
        print(f"\n⚠️  Ensemble slightly underperformed best base model:")
        print(f"   Difference: {improvement:.0f} AED")
        print(f"   Consider using best base model ({best_base['Model']}) instead")
    else:
        print(f"\n✅ Ensemble matches best base model performance")
    
    # Overall journey
    print(f"\n📈 MODEL PERFORMANCE SUMMARY:")
    print(f"   Validation MAE:  {val_mae:,.0f} AED")
    print(f"   Test MAE:        {test_mae:,.0f} AED")
    print(f"   Test R²:         {test_r2:.4f}")
    print(f"   Improvement:     {improvement_pct:+.1f}% over best base model")


if __name__ == "__main__":
    main()

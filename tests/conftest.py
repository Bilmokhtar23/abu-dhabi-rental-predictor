"""
Pytest Configuration and Shared Fixtures
========================================

This file contains shared fixtures used across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'model_outputs' / 'production'


@pytest.fixture(scope="session")
def sample_train_data():
    """
    Returns a 100-row sample from train_set_FINAL.csv for testing.
    Loaded once per test session for performance.
    """
    df = pd.read_csv(DATA_DIR / 'train_set_FINAL.csv')
    return df.sample(n=100, random_state=42)


@pytest.fixture(scope="session")
def sample_val_data():
    """Returns a 50-row sample from validation set"""
    df = pd.read_csv(DATA_DIR / 'val_set_FINAL.csv')
    return df.sample(n=50, random_state=42)


@pytest.fixture(scope="session")
def sample_test_data():
    """Returns a 50-row sample from test set"""
    df = pd.read_csv(DATA_DIR / 'test_set_FINAL.csv')
    return df.sample(n=50, random_state=42)


@pytest.fixture
def sample_input():
    """
    Returns a valid prediction input dictionary.
    Represents a typical 2BR apartment in Al Reem Island.
    """
    return {
        'Location': 'Al Reem Island',
        'Type': 'AP',
        'Furnishing': 'Unfurnished',
        'Beds': 2,
        'Baths': 2,
        'Area_in_sqft': 1200
    }


@pytest.fixture
def sample_input_with_features():
    """
    Returns input with all 14 features pre-computed.
    Useful for testing model prediction directly.
    """
    return {
        'Beds': 2,
        'Baths': 2,
        'Area_in_sqft': 1200,
        'log_area': np.log1p(1200),
        'property_rank_in_location': 0.5,
        'area_deviation_from_location': 0.1,
        'location_type_premium': 1.2,
        'furnishing_type_premium': 1.0,
        'bath_bed_ratio': 2 / 3,
        'area_per_bedroom': 1200 / 3,
        'type_room_premium': 0.8,
        'Location': 'Al Reem Island',
        'Type': 'AP',
        'Furnishing': 'Unfurnished'
    }


@pytest.fixture
def expected_feature_columns():
    """Returns the expected RAW feature columns (14 total before encoding)"""
    # Raw features in the CSV files BEFORE one-hot encoding
    return [
        # Numerical features (11)
        'Beds', 'Baths', 'Area_in_sqft', 'log_area', 'property_rank_in_location',
        'area_deviation_from_location', 'location_type_premium', 'furnishing_type_premium',
        'bath_bed_ratio', 'area_per_bedroom', 'type_room_premium',
        # Categorical features (3) - raw, not encoded
        'Location', 'Type', 'Furnishing'
    ]


class StackedEnsemblePredictor:
    """Wrapper for stacked ensemble prediction logic"""
    def __init__(self, model_dict, encoder=None):
        self.base_models = model_dict['base_models']
        self.meta_model = model_dict['meta_model']
        self.encoder = encoder
        
    def predict(self, X):
        """Predict using stacked ensemble logic with proper encoding"""
        # If X is a DataFrame with raw features, encode it properly
        if self.encoder is not None and hasattr(X, 'columns'):
            # Separate numerical and categorical features
            numerical_cols = [col for col in X.columns if col not in ['Location', 'Type', 'Furnishing']]
            categorical_cols = ['Location', 'Type', 'Furnishing']
            
            # Encode categorical features
            categorical_encoded = self.encoder.transform(X[categorical_cols])
            categorical_encoded_df = pd.DataFrame(
                categorical_encoded, 
                columns=self.encoder.get_feature_names_out(categorical_cols),
                index=X.index
            )
            
            # Combine numerical and encoded categorical features
            X_encoded = pd.concat([X[numerical_cols], categorical_encoded_df], axis=1)
        else:
            X_encoded = X

        # 1. Generate predictions from each base model
        base_preds = []
        for name, model in self.base_models.items():
            pred = model.predict(X_encoded)
            base_preds.append(pred)
            
        # 2. Stack base model predictions as meta-features
        meta_features = np.column_stack(base_preds)
        
        # 3. Meta-model makes final prediction
        return self.meta_model.predict(meta_features)

@pytest.fixture(scope="session")
def production_model():
    """
    Loads the production stacked ensemble model.
    Returns a StackedEnsemblePredictor instance that mimics app.py logic.
    """
    try:
        ensemble_dict = joblib.load(MODEL_DIR / 'stacked_ensemble_latest.joblib')
        
        # Try to get encoder from ensemble or file
        encoder = None
        if 'encoder' in ensemble_dict:
            encoder = ensemble_dict['encoder']
        else:
            encoder_path = MODEL_DIR / 'target_encoder_latest.joblib'
            if encoder_path.exists():
                encoder = joblib.load(encoder_path)
                
        return StackedEnsemblePredictor(ensemble_dict, encoder)
    except FileNotFoundError:
        pytest.skip("Production model not found")


@pytest.fixture(scope="session")
def target_encoder():
    """Loads the production target encoder"""
    try:
        # First try loading from ensemble dict
        ensemble_dict = joblib.load(MODEL_DIR / 'stacked_ensemble_latest.joblib')
        if 'encoder' in ensemble_dict:
            return ensemble_dict['encoder']
        # Fallback to separate file
        encoder = joblib.load(MODEL_DIR / 'target_encoder_latest.joblib')
        return encoder
    except FileNotFoundError:
        pytest.skip("Target encoder not found")


@pytest.fixture(scope="session")
def feature_metadata():
    """Loads feature metadata JSON"""
    try:
        with open(DATA_DIR / 'final_feature_metadata.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.skip("Feature metadata not found")


@pytest.fixture(scope="session")
def split_metadata():
    """Loads train/val/test split metadata"""
    try:
        with open(DATA_DIR / 'split_metadata_FINAL.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.skip("Split metadata not found")


@pytest.fixture
def edge_case_inputs():
    """
    Returns list of edge case inputs for robustness testing.
    Tests model behavior on unusual but valid inputs.
    """
    return [
        # Studio apartment (0 beds)
        {
            'Location': 'Al Reem Island',
            'Type': 'AP',
            'Furnishing': 'Furnished',
            'Beds': 0,
            'Baths': 1,
            'Area_in_sqft': 450
        },
        # Large villa
        {
            'Location': 'Saadiyat Island',
            'Type': 'VH',
            'Furnishing': 'Furnished',
            'Beds': 7,
            'Baths': 8,
            'Area_in_sqft': 8000
        },
        # Minimum area
        {
            'Location': 'Al Reef',
            'Type': 'AP',
            'Furnishing': 'Unfurnished',
            'Beds': 1,
            'Baths': 1,
            'Area_in_sqft': 400
        },
        # Maximum reasonable area
        {
            'Location': 'Yas Island',
            'Type': 'TH',
            'Furnishing': 'Semi-Furnished',
            'Beds': 5,
            'Baths': 6,
            'Area_in_sqft': 5000
        }
    ]


@pytest.fixture
def baseline_stats():
    """
    Returns baseline statistics from training data.
    Used for drift detection testing.
    """
    df = pd.read_csv(DATA_DIR / 'train_set_FINAL.csv')
    
    numeric_cols = [
        'Beds', 'Baths', 'Area_in_sqft', 'log_area',
        'property_rank_in_location', 'area_deviation_from_location',
        'location_type_premium', 'furnishing_type_premium',
        'bath_bed_ratio', 'area_per_bedroom', 'type_room_premium'
    ]
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'q25': float(df[col].quantile(0.25)),
            'q50': float(df[col].quantile(0.50)),
            'q75': float(df[col].quantile(0.75))
        }
    
    return stats


@pytest.fixture
def mock_predictions():
    """
    Returns mock prediction data for monitoring tests.
    Simulates a batch of predictions.
    """
    return pd.DataFrame({
        'timestamp': pd.date_range('2025-11-01', periods=100, freq='h'),
        'prediction': np.random.normal(100000, 20000, 100),
        'Location': np.random.choice(['Al Reem Island', 'Yas Island', 'Saadiyat Island'], 100),
        'Type': np.random.choice(['AP', 'VH', 'TH'], 100),
        'Area_in_sqft': np.random.randint(500, 3000, 100),
        'Beds': np.random.randint(1, 5, 100)
    })


@pytest.fixture(autouse=True)
def cleanup_test_artifacts(request):
    """
    Automatically cleanup test artifacts after each test.
    Removes temporary files created during testing.
    """
    yield
    # Cleanup code runs after test
    test_db = PROJECT_ROOT / 'tests' / 'test_predictions.db'
    if test_db.exists():
        test_db.unlink()

"""
Model Configuration
Central configuration for model training and prediction
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'model_outputs'
MLFLOW_DIR = PROJECT_ROOT / 'mlruns'

# Data paths
RAW_DATA_PATH = DATA_DIR / 'raw' / 'abudhabi_properties_cleaned.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed'
TRAIN_SET_PATH = PROCESSED_DATA_PATH / 'train_set_FINAL.csv'
VAL_SET_PATH = PROCESSED_DATA_PATH / 'val_set_FINAL.csv'
TEST_SET_PATH = PROCESSED_DATA_PATH / 'test_set_FINAL.csv'

# Model paths
PRODUCTION_MODEL_DIR = MODEL_DIR / 'production'

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "abu_dhabi_rental_prediction"

# Data cleaning parameters
DATA_CLEANING = {
    'price_min': 25000,
    'price_max': 450000,
    'size_min': 250,
    'size_max': 5000,
    'price_per_sqft_min': 40,
    'price_per_sqft_max': 300,
    'bedrooms_max': 6,
    'bathrooms_max': 8
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'production': {
        'n_features': 14,
        'numeric_features': [
            'Beds', 'Baths', 'Area_in_sqft', 'log_area',
            'property_rank_in_location', 'area_deviation_from_location',
            'location_type_premium', 'furnishing_type_premium',
            'bath_bed_ratio', 'area_per_bedroom', 'type_room_premium'
        ],
        'categorical_features': [
            'Location', 'Type', 'Furnishing'
        ],
        'all_features': [
            'Beds', 'Baths', 'Area_in_sqft', 'log_area',
            'property_rank_in_location', 'area_deviation_from_location',
            'location_type_premium', 'furnishing_type_premium',
            'bath_bed_ratio', 'area_per_bedroom', 'type_room_premium',
            'Location', 'Type', 'Furnishing'
        ],
        'target_encoder_smoothing': 10.0
    }
}

# Model hyperparameters
MODEL_PARAMS = {
    'production': {
        'train_size': 0.70,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_state': 42,
        'cv_folds': 5,

        'xgboost': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },

        'lightgbm': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },

        'catboost': {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.05,
            'random_state': 42,
            'verbose': False
        },

        'ridge_meta': {
            'alpha': 100.0
        },

        'target_encoder': {
            'smoothing': 10.0,
            'handle_unknown': 'value'
        }
    }
}

# Validation schema
VALIDATION_SCHEMA = {
    'price': {
        'min': 10000,
        'max': 1000000,
        'expected_min': 25000,
        'expected_max': 450000
    },
    'size': {
        'min': 100,
        'max': 10000,
        'expected_min': 250,
        'expected_max': 5000
    },
    'bedrooms': {
        'min': 0,
        'max': 10
    },
    'bathrooms': {
        'min': 1,
        'max': 10
    },
    'price_per_sqft': {
        'min': 40,
        'max': 300
    }
}

# Web app configuration
WEB_APP_CONFIG = {
    'title': 'Abu Dhabi Rental Price Predictor',
    'model_version': 'Production (Stacked Ensemble)',
    'model_type': 'XGBoost + LightGBM + CatBoost + Ridge',
    'model_date': '2025-11-25',
    'test_r2': 0.8771,
    'test_mae': 14650,
    'test_rmse': 23268
}

# Performance targets
PERFORMANCE_TARGETS = {
    'production': {
        'test_mae_target': 15000,  # AED
        'test_r2_target': 0.85,
        'test_rmse_target': 25000,  # AED
        'val_mae_target': 15500  # AED
    }
}

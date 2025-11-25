# Model Training Guide

## Overview

This guide describes the actual training pipeline used to develop the Abu Dhabi rental price prediction model. The process involves hyperparameter tuning of individual models followed by stacked ensemble training.

## Training Pipeline

### Phase 1: Hyperparameter Tuning

**Script**: `scripts/train_hyperparameter_tuning.py` (not included in repo)

**Process**:
1. Load pre-split FINAL datasets (train/val/test)
2. Perform grid search on validation set for each algorithm
3. Save best parameters to `model_outputs/tuned/best_params_20251123_185430.json`

**Tuned Models**:
- **XGBoost**: 200 estimators, max_depth=6, learning_rate=0.1
- **LightGBM**: 200 estimators, max_depth=10, learning_rate=0.1
- **CatBoost**: 200 estimators, max_depth=8, learning_rate=0.1

### Phase 2: Stacked Ensemble Training

**Script**: `scripts/train_stacked_ensemble.py`

**Process**:
1. Load tuned hyperparameters
2. Train base models with best parameters
3. Generate meta-features using 5-fold cross-validation (out-of-fold predictions)
4. Train Ridge meta-learner on meta-features
5. Evaluate on validation and test sets
6. Save production model artifacts

**Key Features**:
- **Out-of-fold Predictions**: Prevents data leakage in meta-features
- **Cross-validation**: 5-fold CV for robust meta-feature generation
- **Meta-learner Tuning**: Grid search for Ridge alpha (0.1 to 100.0)
- **Comprehensive Evaluation**: Base model and ensemble performance metrics

## Model Performance

### Overall Results
- **Test R²**: 0.9379 (93.79%)
- **Test MAE**: 5,521 AED
- **Test RMSE**: 26,114 AED
- **Improvement**: Slight underperformance vs. best base model (CatBoost)

### Base Model Performance
| Model | Test MAE | Test R² | Notes |
|-------|----------|---------|-------|
| XGBoost | - | - | Tuned hyperparameters |
| LightGBM | - | - | Tuned hyperparameters |
| CatBoost | 5,456 AED | - | Best individual performer |
| Stacked Ensemble | 5,521 AED | 0.9379 | Meta-learning combination |

## Training Artifacts

### Saved Files
- `model_outputs/tuned/best_params_*.json` - Hyperparameter configurations
- `model_outputs/ensemble/stacked_ensemble_*.joblib` - Trained ensemble model
- `model_outputs/ensemble/ensemble_metadata_*.json` - Training metadata
- `model_outputs/production/stacked_ensemble_latest.joblib` - Production model
- `model_outputs/production/feature_columns_latest.json` - Feature order
- `model_outputs/production/model_metadata_latest.json` - Model metadata

### Model Components
- **Base Models**: 3 trained gradient boosting models
- **Meta Model**: Ridge regression (alpha=100.0)
- **Encoder**: OneHotEncoder for categorical features
- **Feature Columns**: 103 encoded feature names
- **Metadata**: Performance metrics and training details

## Reproducing Training

### Prerequisites
```bash
pip install -r requirements.txt
```

### Full Training Pipeline
```bash
# Note: Hyperparameter tuning script not included
# Assumes best_params_*.json exists

# Train stacked ensemble
python scripts/train_stacked_ensemble.py
```

### Expected Output
```
✅ PHASE 4 COMPLETE!
🏆 FINAL STACKED ENSEMBLE:
   Val MAE:   5,921 AED
   Test MAE:  5,521 AED
   Test R²:   0.9379
   Test RMSE: 26,114 AED
```

## Key Technical Details

### Data Leakage Prevention
- Features engineered after train/test split
- Out-of-fold predictions for meta-features
- No information from test set used in training

### Validation Methodology
- Stratified train (66%) / val (14%) / test (20%) split
- Hyperparameter tuning on validation set
- Final evaluation on held-out test set

### Ensemble Architecture
```
Base Models (XGBoost, LightGBM, CatBoost)
    ↓ (Out-of-fold predictions)
Meta-Features (shape: n_samples × 3)
    ↓ (Ridge Regression)
Final Prediction
```

## Performance Analysis

### Strengths
- **High Accuracy**: 93.8% R² explains most rental price variance
- **Robust**: Ensemble reduces overfitting vs. individual models
- **Production-Ready**: Handles real-time inference efficiently

### Limitations
- **Villa Performance**: Higher MAE for villas vs. apartments
- **Location Variance**: Performance varies by neighborhood data availability
- **Feature Scope**: Limited to basic property characteristics

### Areas for Improvement
- Additional villa-specific features
- More granular location encoding
- Temporal market trend features
- External data integration (amenities, transport)

## Troubleshooting

### Common Issues
- **Memory Errors**: Reduce batch size or use smaller models
- **Convergence Issues**: Check hyperparameter ranges
- **Poor Performance**: Verify data preprocessing and feature engineering

### Validation Checks
- Compare train/val/test distributions
- Check for data leakage in features
- Verify cross-validation stability
- Review feature importance consistency

# Production Scripts

**Last updated**: 2025-11-24 18:05:47

This directory contains **production-ready scripts only**. Old versions and one-time fixes have been archived.

## Active Production Scripts

### Data Preparation
- **`create_train_val_test_split.py`** - Creates proper train/validation/test split (70/15/15)
  - Prevents hyperparameter tuning leakage
  - Stratified by location frequency × rent quartile
  - Usage: `python scripts/create_train_val_test_split.py`

- **`create_final_feature_set.py`** - Creates final 14-feature dataset
  - No data leakage (removed Rent_per_sqft, property_value_tier)
  - No multicollinearity (max r < 0.93)
  - Usage: `python scripts/create_final_feature_set.py`

### Model Training
- **`train_ensemble_no_leakage.py`** - Trains production ensemble model
  - Stacked ensemble: XGBoost + LightGBM + CatBoost + Ridge meta-learner
  - Uses clean train/val/test splits
  - No data leakage
  - Usage: `python scripts/train_ensemble_no_leakage.py`

### Monitoring
- **`audit_pipeline.py`** - Audits entire ML pipeline for issues
  - Checks for data leakage
  - Validates feature engineering
  - Reviews model performance
  - Usage: `python scripts/audit_pipeline.py`

## Workflow

### Initial Setup
1. `create_train_val_test_split.py` - Create train/val/test splits
2. `create_final_feature_set.py` - Create final feature set

### Training
3. `train_ensemble_no_leakage.py` - Train production model

### Validation
4. `audit_pipeline.py` - Validate entire pipeline

## Archived Scripts

See `scripts/archive/` for old versions and deprecated scripts.

## Design Principles

All production scripts follow these principles:
1. **No data leakage** - Features never use target variable
2. **Proper validation** - Train/val/test splits never mixed
3. **Reproducibility** - Fixed random seeds (42)
4. **Documentation** - Clear docstrings and comments
5. **Production-ready** - Can be deployed without modification

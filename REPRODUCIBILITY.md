# Reproducibility Guide

## Quick Start (TL;DR)

```bash
# 1. Clone the repository
git clone https://github.com/Bilmokhtar23/abu-dhabi-rental-predictor.git
cd abu-dhabi-rental-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training (optional - model is already trained)
python scripts/train_stacked_ensemble.py

# 4. Launch the web app
streamlit run app.py

# 5. Run tests to verify everything works
pytest tests/ -v
```

---

## Detailed Setup Instructions

### Prerequisites

- **Python 3.11+** (tested on 3.11.8)
- **pip** (latest version recommended)
- **Git** (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Bilmokhtar23/abu-dhabi-rental-predictor.git
cd abu-dhabi-rental-predictor
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n rental-predictor python=3.11
conda activate rental-predictor
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies for basic reproducibility:**
- pandas==2.1.4
- numpy==1.26.4
- scikit-learn==1.3.2
- xgboost==2.0.3
- lightgbm==4.1.0
- catboost==1.2.2
- streamlit>=1.28.0
- joblib>=1.3.0
- pytest>=7.4.0 (for testing)

**Note:** The requirements.txt includes optional dependencies for advanced features (MLflow, FastAPI, monitoring tools). These are NOT required for basic usage.

---

## Reproducing the Model

### Option A: Use Pre-Trained Model (Fastest)

The repository includes a pre-trained model at:
```
model_outputs/production/stacked_ensemble_latest.joblib
```

Simply run the web app:
```bash
streamlit run app.py
```

### Option B: Retrain from Scratch (2-3 minutes)

```bash
python scripts/train_stacked_ensemble.py
```

**Expected output:**
```
Training Stacked Ensemble...
✅ Training complete!
📊 Test R²: 0.9379 (93.79%)
📏 Test MAE: 5,521 AED
📏 Test RMSE: 26,114 AED
```

**What gets created:**
- `model_outputs/production/stacked_ensemble_latest.joblib` - Main ensemble model
- `model_outputs/production/feature_columns_latest.json` - Feature names (103 encoded features)
- `model_outputs/production/model_metadata_latest.json` - Performance metrics

---

## Verifying Reproducibility

### Run Tests

```bash
# Run all tests (should see 63 passed, 3 skipped)
pytest tests/ -v

# Run specific test suites
pytest tests/test_data/ -v          # Data integrity tests
pytest tests/test_models/ -v        # Model prediction tests
pytest tests/test_scripts/ -v       # Training pipeline tests
pytest tests/test_features/ -v      # Feature engineering tests
```

**Expected result:** All 63 tests should pass, 3 monitoring tests should be skipped.

### Verify Model Performance

After training, check the model metadata:
```bash
cat model_outputs/production/model_metadata_latest.json
```

**Expected metrics:**
```json
{
  "test_r2": 0.9379,
  "test_mae": 5521.11,
  "test_rmse": 26113.70
}
```

---

## Data Files

The repository includes processed datasets with NO data leakage:

| File | Rows | Purpose |
|:-----|:-----|:--------|
| `data/processed/train_set_FINAL.csv` | 15,367 | Model training (66%) |
| `data/processed/val_set_FINAL.csv` | 3,283 | Hyperparameter tuning (14%) |
| `data/processed/test_set_FINAL.csv` | 4,663 | Final evaluation (20%) |

**Features in each file:**
- **14 raw features** (11 numeric + 3 categorical)
- Categorical features: Location, Type, Furnishing
- Numeric features: Beds, Baths, Area_in_sqft, log_area, property_rank_in_location, area_deviation_from_location, location_type_premium, furnishing_type_premium, bath_bed_ratio, area_per_bedroom, type_room_premium
- Target variable: Rent (annual rental price in AED)

**Note:** During training, the 3 categorical features are one-hot encoded, creating 103 total features for the model.

---

## Model Architecture

### Stacked Ensemble Components

1. **Base Models** (trained with 5-fold cross-validation):
   - XGBoost Regressor
   - LightGBM Regressor
   - CatBoost Regressor

2. **Meta-Learner**:
   - Ridge Regression (alpha=100.0)
   - Trained on out-of-fold predictions from base models

3. **Feature Encoding**:
   - OneHotEncoder for categorical features (Location, Type, Furnishing)
   - Fitted ONLY on training data to prevent leakage

### Hyperparameters

Pre-tuned hyperparameters are loaded from:
```
model_outputs/tuned/best_params_20251123_185430.json
```

These were optimized using Optuna with 5-fold cross-validation on the validation set.

---

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'xgboost'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue 2: File Not Found Errors

**Problem:** `FileNotFoundError: data/processed/train_set_FINAL.csv`

**Solution:**
Ensure you're running commands from the project root:
```bash
cd /path/to/abu-dhabi-rental-predictor
python scripts/train_stacked_ensemble.py
```

### Issue 3: Different Results

**Problem:** Getting slightly different R² scores

**Explanation:**
- If you see R² around **93.6-93.9%**, this is normal due to:
  - Different library versions (XGBoost, LightGBM, CatBoost)
  - Floating-point precision differences across platforms
  - Minor random seed variations in base models

**Expected range:** R² between 93.5% and 94.0% is acceptable.

### Issue 4: Tests Failing

**Problem:** Tests fail after fresh clone

**Solution:**
1. Ensure you've run training first:
   ```bash
   python scripts/train_stacked_ensemble.py
   ```

2. Check that model files exist:
   ```bash
   ls -lh model_outputs/production/
   ```

3. Re-run tests:
   ```bash
   pytest tests/ -v
   ```

---

## Minimal Reproducibility Checklist

- [ ] Python 3.11+ installed
- [ ] Repository cloned
- [ ] Virtual environment created (recommended)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Training completed successfully (`python scripts/train_stacked_ensemble.py`)
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Web app launches (`streamlit run app.py`)
- [ ] Predictions work in the web app

---

## Performance Benchmarks

### Training Time (on standard laptop)

- **XGBoost**: ~30 seconds
- **LightGBM**: ~20 seconds
- **CatBoost**: ~40 seconds
- **Meta-learner**: <1 second
- **Total training time**: ~2-3 minutes

### Expected Metrics

| Metric | Expected Value | Acceptable Range |
|:-------|:--------------|:-----------------|
| Test R² | 0.9379 | 0.935 - 0.940 |
| Test MAE | 5,521 AED | 5,400 - 5,650 AED |
| Test RMSE | 26,114 AED | 25,800 - 26,400 AED |
| Val MAE | 5,921 AED | 5,800 - 6,050 AED |

---

## Data Integrity Checks

The repository includes comprehensive tests to ensure data quality:

```bash
# Check for data leakage
pytest tests/test_data/test_loader.py::TestNoDataLeakage -v

# Verify split ratios
pytest tests/test_data/test_loader.py::TestDataSplits -v

# Confirm feature consistency
pytest tests/test_data/test_loader.py::TestFeatureConsistency -v
```

---

## File Locations

### Required for Basic Usage
```
data/processed/
├── train_set_FINAL.csv          # 15,367 rows × 15 columns (14 features + target)
├── val_set_FINAL.csv            # 3,283 rows × 15 columns
├── test_set_FINAL.csv           # 4,663 rows × 15 columns
├── split_metadata_FINAL.json    # Split metadata
└── final_feature_metadata.json  # Feature metadata

model_outputs/production/
├── stacked_ensemble_latest.joblib      # Main model (includes encoder)
├── feature_columns_latest.json         # 103 encoded feature names
└── model_metadata_latest.json          # Performance metrics

model_outputs/tuned/
└── best_params_20251123_185430.json    # Hyperparameters
```

### Optional (for advanced usage)
```
src/                    # Source code modules (for API development)
docs/                   # Development journey and MLflow guide
tests/                  # Comprehensive test suite
```

---

## Contact and Support

If you encounter reproducibility issues:

1. Check this guide first
2. Verify your Python version: `python --version`
3. Check installed package versions: `pip list`
4. Open an issue on GitHub with:
   - Python version
   - Operating system
   - Error message
   - Steps to reproduce

---

**Last Updated:** November 25, 2025
**Tested On:** macOS (Darwin 25.1.0), Python 3.11.8

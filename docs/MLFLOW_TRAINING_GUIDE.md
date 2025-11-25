# MLflow Model Training Guide

## Quick Start

### 1. Run All Experiments

```bash
cd "/Users/bilal/Property finder"
python scripts/train_with_mlflow.py
```

This will:
- Load the optimized dataset (23,313 properties)
- Remove outliers (3×IQR method)
- Train 8 models across 3 experiment groups
- Track all experiments in MLflow
- Save comparison results

### 2. View MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open: http://localhost:5000

### 3. Compare Results

Results saved to: `model_outputs/production/model_comparison.csv`

## Models Trained

### Baseline Models (3)
1. **LinearRegression** - Simple linear model with scaling
2. **Ridge** - L2 regularized linear model  
3. **DecisionTree** - Single decision tree (max_depth=20)

### Ensemble Models (3)
4. **RandomForest** - 200 trees, depth=20
5. **GradientBoosting** - 200 estimators, depth=10
6. **XGBoost** - 200 estimators, depth=10, learning_rate=0.1

### Advanced Models (2)
7. **LightGBM** - Fast gradient boosting, 200 estimators
8. **CatBoost** - Gradient boosting with categorical support

## What's Tracked in MLflow

### Parameters
- `model_type` - Model algorithm name
- `experiment_group` - baseline/ensemble/advanced
- `n_features` - Number of features (27)
- `train_size` / `test_size` - Dataset sizes
- `use_scaling` - Whether StandardScaler applied
- Model-specific hyperparameters (n_estimators, max_depth, etc.)

### Metrics
- `test_mae` - Mean Absolute Error on test set
- `test_r2` - R² score on test set  
- `test_rmse` - Root Mean Squared Error
- `test_mape` - Mean Absolute Percentage Error
- `cv_mae_mean` / `cv_mae_std` - Cross-validation MAE
- `cv_r2_mean` / `cv_r2_std` - Cross-validation R²
- `training_time` - Training duration (seconds)
- `inference_time_ms` - Prediction time per sample
- `mae_beds_X` - MAE by bedroom count
- `mae_price_low/medium/high/luxury` - MAE by price range

### Artifacts
- `model` - Trained scikit-learn model
- `temp_*_visualizations.png` - 4-panel visualization:
  - Predictions vs Actual scatter
  - Residual plot
  - Residual distribution
  - Feature importance (top 15)
- `temp_*_feature_importance.csv` - Full feature importance scores
- `temp_*_features.json` - List of features used
- `temp_*_scaler.pkl` - StandardScaler (if used)

## Expected Performance Targets

- **MAE**: < 5,000 AED
- **R²**: ≥ 0.85
- **Training Time**: < 60s per model
- **Inference Time**: < 10ms per sample

## Interpreting Results

### Best Model Selection
Models ranked by MAE (lowest = best)

Check:
1. **Test MAE** - Primary metric for accuracy
2. **R²** - Variance explained (higher = better)
3. **CV MAE std** - Lower = more stable
4. **Training Time** - Production deployment consideration
5. **Feature Importance** - Model interpretability

### Segment Performance
Review MAE by:
- **Bedroom count** - Performance across property sizes
- **Price ranges** - Accuracy for low/medium/high/luxury segments

Good model should have:
- Consistent performance across segments
- Lower error on common property types (1-3BR, medium price)

## Next Steps After Training

### 1. Review Results
```bash
# View comparison table
cat model_outputs/production/model_comparison.csv

# Launch MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

### 2. Register Best Model
```python
import mlflow

# Get best run ID from comparison.csv
best_run_id = "YOUR_RUN_ID"

# Register model
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "abu_dhabi_rental_predictor")
```

### 3. Hyperparameter Tuning (Optional)
If best model doesn't meet targets, run grid search:
```python
# See scripts/train_production_model.py for GridSearchCV example
```

### 4. Deploy to Production
```python
# Load registered model
model = mlflow.pyfunc.load_model("models:/abu_dhabi_rental_predictor/Production")

# Make predictions
predictions = model.predict(X_new)
```

## Troubleshooting

### MLflow UI not loading
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

### Models training slowly
- Check CPU usage (n_jobs=-1 uses all cores)
- Reduce n_estimators or max_depth
- Use fewer CV folds (cv=3 instead of 5)

### Poor performance (MAE > 10,000)
- Check data quality (outliers removed?)
- Verify feature engineering (27 features expected)
- Review feature importance (top features make sense?)
- Try different hyperparameters

## Configuration

Edit in `scripts/train_with_mlflow.py`:

```python
# Configuration
DATA_PATH = "data/processed/abudhabi_properties_OPTIMIZED.csv"
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 3.0  # IQR multiplier
TEST_SIZE = 0.2          # 20% test set
RANDOM_STATE = 42        # Reproducibility
```

## Dataset Requirements

Expected structure:
- **27 numeric features**: Area_in_sqft, Beds, Baths, room_count, etc.
- **3 categorical** (not used in training): Location, Type, Furnishing
- **1 target**: Rent (AED)

No missing or infinite values allowed.

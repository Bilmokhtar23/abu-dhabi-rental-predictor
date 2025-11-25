# 🏠 Abu Dhabi Rental Price Predictor

**Production-Ready Real Estate Valuation System**

A machine learning system for predicting rental prices in Abu Dhabi. This project demonstrates a complete end-to-end ML pipeline, from data processing and feature engineering to ensemble modeling and production deployment via an interactive web application.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
![R² Score](https://img.shields.io/badge/R²-93.8%25-blue.svg)

---

## 🌟 Executive Summary

This system analyzes over **23,000 properties** in Abu Dhabi to provide accurate rental valuations. By leveraging a **Stacked Ensemble** of XGBoost, LightGBM, and CatBoost with Ridge meta-learner, the model achieves a **93.8% R² score** on held-out test data with proper validation methodology.

### Key Achievements
- **🎯 Accuracy**: Test R² of 93.8%, explaining variance in rental prices across diverse property types
- **📊 Precision**: Mean Absolute Error of 5,521 AED (RMSE: 26,114 AED)
- **🧠 Clean Methodology**: No data leakage, proper train/validation/test splits with stratification
- **🏗️ Feature Engineering**: 14 carefully selected features (11 numeric + 3 categorical)
- **🌐 Production App**: Streamlit interface for real-time predictions with price comparisons

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip (latest version)

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/Bilmokhtar23/abu-dhabi-rental-predictor.git
cd abu-dhabi-rental-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (optional - takes ~2-3 minutes)
python scripts/train_stacked_ensemble.py

# 4. Launch the web app
streamlit run app.py

# 5. Run tests to verify setup
pytest tests/ -v
```

The application will open in your browser at `http://localhost:8501`.

**For detailed reproducibility instructions, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).**

---

## 📊 Data

The repository includes processed datasets for immediate reproducibility:

- **train_set_FINAL.csv**: 15,367 properties for model training
- **val_set_FINAL.csv**: 3,283 properties for hyperparameter tuning
- **test_set_FINAL.csv**: 4,663 properties for final evaluation

Data includes 14 raw features (11 numeric + 3 categorical) that are encoded to 103 features for model training.

---

## 📊 System Architecture

The project follows a production-ready ML pipeline:

1.  **Data Ingestion**: Processing raw property data (23,281 records from Abu Dhabi market).
2.  **Preprocessing**: Stratified train/validation/test split (70/15/15) to prevent data leakage.
3.  **Feature Engineering**:
    *   **Domain Features**: Location rankings, area deviations, property type premiums
    *   **Derived Features**: Log-transformed area, bath-to-bed ratios, area per bedroom
    *   **Target Encoding**: Smooth categorical encoding (Location, Type, Furnishing) with 10.0 smoothing factor to prevent overfitting
4.  **Model Training**:
    *   **Base Models**: XGBoost, LightGBM, CatBoost (individually tuned with hyperparameter optimization)
    *   **Meta-Features**: Out-of-fold predictions using 5-fold cross-validation to prevent leakage
    *   **Meta Learner**: Ridge Regression (alpha=100.0) combines base model outputs
5.  **Deployment**: Streamlit app loads production ensemble artifacts for real-time inference.

---

## 📱 Web Application Features

### 🔮 Price Predictor
*   **Instant Valuations**: Enter property details to get an immediate rental estimate
*   **Market Comparison**: Compare predictions against location and property type averages
*   **Clean Interface**: Simple, user-friendly Streamlit interface for quick valuations

---

## 📂 Project Structure

```text
Property finder/
├── app.py                          # Main Streamlit Web Application
├── requirements.txt                # Project dependencies
├── data/
│   ├── raw/
│   │   └── abudhabi_properties_cleaned.csv  # Raw dataset (23,313 properties)
│   └── processed/
│       ├── train_set_FINAL.csv     # Training set (15,367 properties)
│       ├── val_set_FINAL.csv       # Validation set (3,283 properties)
│       ├── test_set_FINAL.csv      # Test set (4,663 properties)
│       └── *.json                  # Feature and split metadata
├── model_outputs/
│   └── production/
│       ├── stacked_ensemble_latest.joblib    # Production ensemble model
│       ├── target_encoder_latest.joblib      # Categorical encoder
│       ├── xgboost_latest.joblib             # Base XGBoost model
│       └── *.json                            # Model metadata
├── scripts/
│   ├── train_stacked_ensemble.py   # Main training pipeline
│   ├── apply_final_feature_set.py  # Feature engineering documentation
│   └── audit_pipeline.py           # Pipeline validation tool
├── docs/
│   ├── DEVELOPMENT_JOURNEY.md      # Development narrative and insights
│   └── MLFLOW_TRAINING_GUIDE.md    # MLflow tracking guide
└── src/                            # Source code modules (future API)
```

---

## 📈 Model Performance

| Metric | Value | Description |
|:-------|:------|:------------|
| **R² Score** | **0.9379** | Explains 93.8% of rental price variance |
| **MAE** | **5,521 AED** | Mean Absolute Error |
| **RMSE** | **26,114 AED** | Root Mean Squared Error |
| **Validation MAE** | **5,921 AED** | Validation set performance |

*Performance measured on held-out test set of 4,663 properties (20% of dataset).*

### Model Details
- **Training Data**: 15,367 properties (66% split)
- **Validation Data**: 3,283 properties (14% split)
- **Test Data**: 4,663 properties (20% split)
- **Features**: 14 total (11 numeric + 3 categorical)
- **Training Date**: November 25, 2025
- **Base Models**: XGBoost, LightGBM, CatBoost
- **Meta-Learner**: Ridge Regression (alpha=100.0)

---

## 🛠️ Technologies Used

*   **Core**: Python 3.11, Pandas, NumPy
*   **Machine Learning**: XGBoost, LightGBM, CatBoost, Scikit-Learn, Category Encoders
*   **Visualization**: Plotly Express, Seaborn, Matplotlib
*   **Web Framework**: Streamlit, Streamlit-Folium
*   **Geospatial**: Folium
*   **Utilities**: Joblib (model serialization)

---

## 📚 Additional Documentation

- **[Reproducibility Guide](REPRODUCIBILITY.md)** - Detailed setup instructions and troubleshooting for external users
- **[Development Journey](docs/DEVELOPMENT_JOURNEY.md)** - Complete narrative of model development, from initial experiments to production deployment
- **[MLflow Training Guide](docs/MLFLOW_TRAINING_GUIDE.md)** - Guide for experiment tracking and model versioning

---

**Author**: Bilal  
**Date**: November 2025  
**License**: MIT (see LICENSE file)

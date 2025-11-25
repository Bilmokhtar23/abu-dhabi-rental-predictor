# Test Configuration

This directory contains comprehensive unit tests for the Abu Dhabi Rental Price Predictor.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=scripts --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_data/test_loader.py -v

# Run specific test class
pytest tests/test_models/test_predictor.py::TestModelPrediction -v

# Run specific test
pytest tests/test_data/test_loader.py::TestDataIntegrity::test_row_counts_match_actual -v
```

## Coverage Goals

- **Overall:** >80%
- **Critical paths:** >90% (data loading, prediction, feature engineering)
- **Model training:** >75%

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── test_data/
│   └── test_loader.py         # Data loading & integrity (18 tests)
├── test_features/
│   └── test_feature_engineering.py  # Feature calculations (13 tests)
├── test_models/
│   └── test_predictor.py      # Model prediction (16 tests)
├── test_scripts/
│   └── test_training.py       # Training pipeline (11 tests)
├── test_monitoring/
│   └── test_monitoring.py     # Monitoring (future)
└── test_app.py                # Streamlit app (6 tests)
```

## Test Categories

- **Data Integrity:** 18 tests
- **Feature Engineering:** 13 tests  
- **Model Prediction:** 16 tests
- **Training Pipeline:** 11 tests
- **App Logic:** 6 tests

**Total: 64 tests**

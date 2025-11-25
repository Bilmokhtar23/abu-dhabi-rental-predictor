"""
Tests for Model Prediction
===========================

Tests for loading models and making predictions.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class TestModelLoading:
    """Test suite for loading model artifacts"""
    
    def test_production_model_exists(self):
        """Verify production model file exists"""
        model_path = Path('model_outputs/production/stacked_ensemble_latest.joblib')
        assert model_path.exists(), "Production model file not found"
        
    def test_production_model_loads(self, production_model):
        """Verify production model loads without errors"""
        assert production_model is not None
        
    def test_target_encoder_exists(self):
        """Verify target encoder file exists"""
        encoder_path = Path('model_outputs/production/target_encoder_latest.joblib')
        assert encoder_path.exists(), "Target encoder file not found"
        
    def test_target_encoder_loads(self, target_encoder):
        """Verify target encoder loads without errors"""
        assert target_encoder is not None
        
    def test_feature_columns_json_exists(self):
        """Verify feature columns JSON exists"""
        feature_path = Path('model_outputs/production/feature_columns_latest.json')
        assert feature_path.exists(), "Feature columns JSON not found"


class TestModelPrediction:
    """Test suite for model prediction functionality"""
    
    def test_predict_on_single_row(self, production_model, sample_test_data):
        """Verify model can predict on single row"""
        X = sample_test_data.drop('Rent', axis=1).iloc[0:1]
        
        prediction = production_model.predict(X)
        
        assert prediction is not None
        assert len(prediction) == 1
        assert isinstance(prediction[0], (int, float, np.number))
        
    def test_predict_on_batch(self, production_model, sample_test_data):
        """Verify model can predict on batch of rows"""
        X = sample_test_data.drop('Rent', axis=1).head(10)
        
        predictions = production_model.predict(X)
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        
    def test_predictions_in_reasonable_range(self, production_model, sample_test_data):
        """Verify predictions are in reasonable price range"""
        X = sample_test_data.drop('Rent', axis=1).head(50)
        
        predictions = production_model.predict(X)
        
        # Predictions should be between 10k and 1M AED (reasonable for Abu Dhabi)
        assert all(10000 <= p <= 1000000 for p in predictions), \
            f"Predictions outside range: {predictions.min():.0f} - {predictions.max():.0f}"
            
    def test_predictions_are_deterministic(self, production_model, sample_test_data):
        """Verify same input produces same prediction"""
        X = sample_test_data.drop('Rent', axis=1).iloc[0:1]
        
        pred1 = production_model.predict(X)
        pred2 = production_model.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2, err_msg="Predictions should be deterministic")
        
    def test_no_nan_predictions(self, production_model, sample_test_data):
        """Verify model never returns NaN predictions"""
        X = sample_test_data.drop('Rent', axis=1)
        
        predictions = production_model.predict(X)
        
        assert not np.isnan(predictions).any(), "Model returned NaN predictions"
        
    def test_no_inf_predictions(self, production_model, sample_test_data):
        """Verify model never returns infinite predictions"""
        X = sample_test_data.drop('Rent', axis=1)
        
        predictions = production_model.predict(X)
        
        assert not np.isinf(predictions).any(), "Model returned infinite predictions"


class TestPredictionAccuracy:
    """Test suite for prediction accuracy on test set"""
    
    def test_model_r2_above_threshold(self, production_model, sample_test_data):
        """Verify R² score on test sample is reasonable"""
        X = sample_test_data.drop('Rent', axis=1)
        y = sample_test_data['Rent']
        
        predictions = production_model.predict(X)
        
        # Calculate R²
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Should be at least 0.70 on sample
        assert r2 >= 0.70, f"R² too low: {r2:.3f}"
        
    def test_predictions_correlate_with_actuals(self, production_model, sample_test_data):
        """Verify predictions are positively correlated with actual values"""
        X = sample_test_data.drop('Rent', axis=1)
        y = sample_test_data['Rent']
        
        predictions = production_model.predict(X)
        
        correlation = np.corrcoef(y, predictions)[0, 1]
        
        # Strong positive correlation expected
        assert correlation >= 0.85, f"Correlation too low: {correlation:.3f}"


class TestEdgeCases:
    """Test suite for edge case handling"""
    
    def test_studio_apartment_prediction(self, production_model):
        """Verify model handles studio (0 beds) correctly"""
        # Create studio apartment input
        studio_data = pd.DataFrame({
            'Beds': [0],
            'Baths': [1],
            'Area_in_sqft': [450],
            'log_area': [np.log1p(450)],
            'property_rank_in_location': [0.5],
            'area_deviation_from_location': [0.0],
            'location_type_premium': [1.0],
            'furnishing_type_premium': [1.0],
            'bath_bed_ratio': [1.0],  # 1 / (0 + 1)
            'area_per_bedroom': [450.0],  # 450 / (0 + 1)
            'type_room_premium': [0.8],
            'Location': [1.0],  # Encoded value
            'Type': [0.5],  # Encoded value
            'Furnishing': [0.3]  # Encoded value
        })
        
        prediction = production_model.predict(studio_data)
        
        assert prediction[0] > 0, "Studio prediction should be positive"
        assert 20000 <= prediction[0] <= 200000, \
            f"Studio prediction {prediction[0]:.0f} seems unreasonable"
            
    def test_large_villa_prediction(self, production_model):
        """Verify model handles large villa correctly"""
        villa_data = pd.DataFrame({
            'Beds': [7],
            'Baths': [8],
            'Area_in_sqft': [8000],
            'log_area': [np.log1p(8000)],
            'property_rank_in_location': [0.9],
            'area_deviation_from_location': [2.0],
            'location_type_premium': [1.5],
            'furnishing_type_premium': [1.2],
            'bath_bed_ratio': [8.0 / 8],  # 8 / (7 + 1)
            'area_per_bedroom': [8000.0 / 8],  # 8000 / (7 + 1)
            'type_room_premium': [1.5],
            'Location': [1.5],  # Encoded value
            'Type': [1.2],  # Encoded value
            'Furnishing': [1.1]  # Encoded value
        })
        
        prediction = production_model.predict(villa_data)
        
        assert prediction[0] > 0, "Villa prediction should be positive"
        assert 100000 <= prediction[0] <= 1000000, \
            f"Villa prediction {prediction[0]:.0f} seems unreasonable"


class TestFeatureImportance:
    """Test suite for feature importance (if available)"""
    
    def test_feature_importance_exists(self):
        """Check if feature importance was saved"""
        # This is optional - test will pass even if not found
        importance_path = Path('model_outputs/production/feature_importance.json')
        
        if importance_path.exists():
            import json
            with open(importance_path, 'r') as f:
                importance = json.load(f)
            assert len(importance) > 0, "Feature importance should not be empty"
        else:
            pytest.skip("Feature importance not saved (optional)")

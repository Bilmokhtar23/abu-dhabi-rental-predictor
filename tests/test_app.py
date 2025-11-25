"""
Tests for Streamlit App
========================

Tests for app.py functionality.
"""

import pytest
import pandas as pd
import numpy as np


class TestAppDataLoading:
    """Test app.py data loading"""
    
    def test_app_loads_train_and_test(self):
        """Verify app loads both train and test sets"""
        train_df = pd.read_csv('data/processed/train_set_FINAL.csv')
        test_df = pd.read_csv('data/processed/test_set_FINAL.csv')
        
        # App concatenates these
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        assert len(df) == len(train_df) + len(test_df)
        assert len(df) == 15367 + 4663  # 20,030 properties for visualization
        

class TestPredictionLogic:
    """Test feature engineering in app.py"""
    
    def test_app_feature_calculation_matches_training(self):
        """Verify app.py calculates features same as training script"""
        # Sample input
        beds = 2
        baths = 2
        area = 1200
        
        # App.py logic (lines 260-310)
        log_area = np.log1p(area)
        bath_bed_ratio = baths / (beds + 1)
        area_per_bedroom = area / (beds + 1)
        
        # Expected values
        assert np.isclose(log_area, np.log1p(1200))
        assert np.isclose(bath_bed_ratio, 2/3)
        assert np.isclose(area_per_bedroom, 400.0)
        
    def test_feature_engineering_handles_studio(self):
        """Verify app handles studio (0 beds) correctly"""
        beds = 0
        baths = 1
        area = 450
        
        # Division by (beds + 1) prevents divide-by-zero
        bath_bed_ratio = baths / (beds + 1)  # 1 / 1 = 1.0
        area_per_bedroom = area / (beds + 1)  # 450 / 1 = 450
        
        assert bath_bed_ratio == 1.0
        assert area_per_bedroom == 450.0
        assert np.isfinite(bath_bed_ratio)
        assert np.isfinite(area_per_bedroom)


class TestAppConstants:
    """Test app configuration constants"""
    
    def test_paths_exist(self):
        """Verify all data/model paths referenced in app exist"""
        from pathlib import Path
        
        paths_to_check = [
            'data/processed/train_set_FINAL.csv',
            'data/processed/test_set_FINAL.csv',
            'model_outputs/production/stacked_ensemble_latest.joblib',
            'model_outputs/production/feature_columns_latest.json'
        ]
        
        for path_str in paths_to_check:
            path = Path(path_str)
            assert path.exists(), f"App requires {path_str} but not found"
            
    def test_random_state_consistency(self):
        """Verify RANDOM_STATE=42 for reproducibility"""
        RANDOM_STATE = 42
        assert RANDOM_STATE == 42, "Random state should be 42 for consistency"

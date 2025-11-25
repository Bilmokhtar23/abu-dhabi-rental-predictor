import pytest
import pandas as pd
import numpy as np
from src.utils.feature_engineering import create_log_features, create_interaction_features

class TestFeatureUtils:
    """Test suite for feature engineering utilities"""
    
    def test_create_log_features(self):
        """Verify log transformation utility"""
        df = pd.DataFrame({
            'A': [1, 10, 100],
            'B': [0, 5, 50]
        })
        
        result = create_log_features(df, ['A', 'B'])
        
        assert 'log_A' in result.columns
        assert 'log_B' in result.columns
        assert np.allclose(result['log_A'], np.log1p(df['A']))
        assert np.allclose(result['log_B'], np.log1p(df['B']))
        
        # Verify original columns untouched
        assert 'A' in result.columns
        assert 'B' in result.columns
        
    def test_create_interaction_features(self):
        """Verify interaction feature creation"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = create_interaction_features(df, [('A', 'B')])
        
        assert 'A_B_interaction' in result.columns
        assert np.array_equal(result['A_B_interaction'], df['A'] * df['B'])

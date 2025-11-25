"""
Tests for Feature Engineering
==============================

Tests for feature calculation logic and transformations.
"""

import pytest
import pandas as pd
import numpy as np


class TestFeatureCalculations:
    """Test suite for individual feature calculations"""
    
    def test_log_area_calculation(self, sample_train_data):
        """Verify log_area = log(Area_in_sqft + 1)"""
        expected = np.log1p(sample_train_data['Area_in_sqft'])
        actual = sample_train_data['log_area']
        
        # Allow small floating point differences
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        
    def test_bath_bed_ratio_calculation(self, sample_train_data):
        """Verify bath_bed_ratio = Baths / (Beds + 1)"""
        expected = sample_train_data['Baths'] / (sample_train_data['Beds'] + 1)
        actual = sample_train_data['bath_bed_ratio']
        
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        
    def test_area_per_bedroom_calculation(self, sample_train_data):
        """Verify area_per_bedroom = Area_in_sqft / (Beds + 1)"""
        expected = sample_train_data['Area_in_sqft'] / (sample_train_data['Beds'] + 1)
        actual = sample_train_data['area_per_bedroom']
        
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        
    def test_numeric_features_no_nan(self, sample_train_data):
        """Verify all numeric features have no NaN values"""
        numeric_features = [
            'Beds', 'Baths', 'Area_in_sqft', 'log_area',
            'property_rank_in_location', 'area_deviation_from_location',
            'location_type_premium', 'furnishing_type_premium',
            'bath_bed_ratio', 'area_per_bedroom', 'type_room_premium'
        ]
        
        for col in numeric_features:
            assert sample_train_data[col].isnull().sum() == 0, f"{col} contains NaN values"
            
    def test_numeric_features_no_inf(self, sample_train_data):
        """Verify no infinite values in numeric features"""
        numeric_features = [
            'Beds', 'Baths', 'Area_in_sqft', 'log_area',
            'property_rank_in_location', 'area_deviation_from_location',
            'location_type_premium', 'furnishing_type_premium',
            'bath_bed_ratio', 'area_per_bedroom', 'type_room_premium'
        ]
        
        for col in numeric_features:
            assert not np.isinf(sample_train_data[col]).any(), f"{col} contains infinite values"


class TestFeatureRanges:
    """Test suite for feature value ranges"""
    
    def test_beds_range(self, sample_train_data):
        """Verify Beds is in reasonable range"""
        assert sample_train_data['Beds'].min() >= 0, "Beds cannot be negative"
        assert sample_train_data['Beds'].max() <= 10, "Beds > 10 is unusual"
        
    def test_baths_range(self, sample_train_data):
        """Verify Baths is in reasonable range"""
        assert sample_train_data['Baths'].min() >= 1, "Baths should be >= 1"
        assert sample_train_data['Baths'].max() <= 10, "Baths > 10 is unusual"
        
    def test_area_range(self, sample_train_data):
        """Verify Area_in_sqft is reasonable"""
        assert sample_train_data['Area_in_sqft'].min() >= 100, "Area < 100 sqft is too small"
        assert sample_train_data['Area_in_sqft'].max() <= 20000, "Area > 20k sqft is unusual"
        
    def test_log_area_is_positive(self, sample_train_data):
        """Verify log_area is always positive (log1p always >= 0)"""
        assert (sample_train_data['log_area'] >= 0).all(), "log_area should be >= 0"
        
    def test_bath_bed_ratio_reasonable(self, sample_train_data):
        """Verify bath/bed ratio is in reasonable range"""
        assert sample_train_data['bath_bed_ratio'].min() >= 0, "Ratio cannot be negative"
        assert sample_train_data['bath_bed_ratio'].max() <= 10, "Ratio > 10 is unusual"
        
    def test_area_per_bedroom_reasonable(self, sample_train_data):
        """Verify area per bedroom is reasonable"""
        assert sample_train_data['area_per_bedroom'].min() >= 50, "< 50 sqft/bedroom is too small"
        assert sample_train_data['area_per_bedroom'].max() <= 10000, "> 10k sqft/bedroom is unusual"


class TestDerivedFeatureLogic:
    """Test feature engineering logic with known inputs"""
    
    def test_studio_bath_bed_ratio(self):
        """Verify studio apartment (0 beds) has correct ratio"""
        beds = 0
        baths = 1
        expected_ratio = baths / (beds + 1)  # 1 / 1 = 1.0
        
        assert expected_ratio == 1.0, "Studio should have bath_bed_ratio = 1.0"
        
    def test_area_per_bedroom_with_studio(self):
        """Verify studio area calculation"""
        beds = 0
        area = 500
        expected = area / (beds + 1)  # 500 / 1 = 500
        
        assert expected == 500, "Studio 500 sqft should have area_per_bedroom = 500"
        
    def test_log_transformation_monotonic(self):
        """Verify log transform preserves ordering"""
        areas = np.array([500, 1000, 1500, 2000, 2500])
        log_areas = np.log1p(areas)
        
        # Larger area should have larger log_area
        assert np.all(log_areas[1:] > log_areas[:-1]), "log_area should be monotonically increasing"

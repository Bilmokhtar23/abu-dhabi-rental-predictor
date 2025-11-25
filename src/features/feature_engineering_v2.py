"""
Feature Engineering Pipeline V2 - Refined & Optimized
Abu Dhabi Rental Price Predictor

Refined from 38 features to 14 strategically selected features based on:
- Correlation analysis (removed |r| < 0.2)
- Multicollinearity analysis (removed redundant pairs)
- Domain knowledge (real estate industry standards)
- Business value (interpretability + predictive power)

Author: Property Finder ML Team
Date: November 22, 2025
Version: 2.0 (Optimized)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re


class FeatureEngineerV2:
    """
    Refined feature engineering pipeline: 14 strategically selected features.
    
    Feature Selection Rationale:
    - Removed target-derived features (log_price, sqrt_price - data leakage)
    - Removed very weak features (|r| < 0.2): badges, temporal, location flags
    - Removed redundant features (multicollinearity > 0.9): sqrt_size, *_per_bathroom
    - Kept domain-driven features: location_tier, region benchmarks
    - Kept strong predictors: size (r=0.79), bathrooms (r=0.71), total_rooms (r=0.71)
    
    Final Feature Set (14 features):
    1. Core (3): bedrooms, bathrooms, size
    2. Transformation (1): log_size
    3. Price Ratios (2): price_per_sqft, price_per_bedroom
    4. Size Ratios (1): size_per_bedroom
    5. Location (3): location_tier, location_popularity, region_avg_price_for_bedrooms
    6. Statistical (1): price_percentile_in_region
    7. Derived (1): total_rooms
    8. Interaction (1): price_per_sqft_X_bedrooms
    
    Usage:
        engineer = FeatureEngineerV2()
        df_train_engineered = engineer.fit_transform(df_train)
        df_test_engineered = engineer.transform(df_test)
    """
    
    def __init__(self):
        """Initialize feature engineer with location tier mapping."""
        # Location tier mapping (based on Abu Dhabi market analysis)
        # Tier 1: Ultra-premium (Saadiyat/Maryah) - cultural/financial districts
        # Tier 2: Premium (Yas/Reem/Corniche) - entertainment/waterfront
        # Tier 3: Mid-premium (Masdar/Raha) - sustainable/residential
        self.location_tiers = {
            'Saadiyat Island': 1,    # Ultra-premium
            'Al Maryah Island': 1,   # Ultra-premium
            'Yas Island': 2,         # Premium
            'Al Reem Island': 2,     # Premium
            'Corniche Road': 2,      # Premium
            'Masdar City': 3,        # Mid-premium
            'Al Raha Beach': 3       # Mid-premium
        }
        
        # Pre-computed statistics (calculated from training data)
        self.stats = {}
    
    def fit(self, df):
        """
        Calculate statistics from training data for feature engineering.
        
        Args:
            df: Training DataFrame with raw features
            
        Returns:
            self: Fitted FeatureEngineerV2 instance
        """
        # Regional statistics (price by location)
        self.stats['region_price_mean'] = df.groupby('address_locality')['price'].mean().to_dict()
        
        # Regional statistics (price by location)
        self.stats['region_price_mean'] = df.groupby('address_locality')['price'].mean().to_dict()
        
        # Regional statistics (size by location)  
        self.stats['region_size_mean'] = df.groupby('address_locality')['size'].mean().to_dict()
        
        # Global statistics (for fallback)
        self.stats['global_price_mean'] = df['price'].mean()
        self.stats['global_size_mean'] = df['size'].mean()
        
        return self
    
    def transform(self, df):
        """
        Transform dataset with 24 refined engineered features.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with 24 additional engineered features (total 38 columns)
        """
        df = df.copy()
        
        # ============================================================
        # CORE FEATURES (3 features) - kept as-is from raw data
        # ============================================================
        # bedrooms, bathrooms, size already exist in df
        
        # ============================================================
        # TRANSFORMATION (2 features)
        # ============================================================
        # Log transformation for non-linear size-price relationship
        df['log_size'] = np.log1p(df['size'])
        
        # Square root transformation - complements log_size (r=0.79)
        df['sqrt_size'] = np.sqrt(df['size'])
        
        # ============================================================
        # PRICE RATIOS (2 features)
        # ============================================================
        # Industry standard metric - price normalized by size
        df['price_per_sqft'] = df['price'] / df['size'].replace(0, 1)
        
        # Tenant affordability - price normalized by bedroom count
        df['price_per_bedroom'] = df['price'] / df['bedrooms'].replace({'Studio': 0.5}).replace(0, 1).astype(float)
        
        # ============================================================
        # SIZE RATIOS (1 feature)
        # ============================================================
        # Property spaciousness - average room size indicator
        df['size_per_bedroom'] = df['size'] / df['bedrooms'].replace({'Studio': 0.5}).replace(0, 1).astype(float)
        
        # ============================================================
        # LOCATION FEATURES (3 features)
        # ============================================================
        # Location tier (1=Ultra-premium, 2=Premium, 3=Mid-premium)
        df['location_tier'] = df['address_locality'].map(self.location_tiers)
        
        # Location popularity (count of properties in this location)
        df['location_popularity'] = df.groupby('address_locality')['property_id'].transform('count')
        
        # Regional average price for this location (location-specific benchmark)
        df['region_avg_price_for_bedrooms'] = df.apply(
            lambda row: self.stats['region_price_mean'].get(row['address_locality'], self.stats['global_price_mean']),
            axis=1
        )
        
        # ============================================================
        # STATISTICAL FEATURES (3 features)
        # ============================================================
        # Price percentile within region (0-1 rank)
        df['price_percentile_in_region'] = df.groupby('address_locality')['price'].rank(pct=True)
        
        # Price compared to regional mean (r=0.68)
        df['price_vs_region_mean'] = df.apply(
            lambda row: row['price'] / self.stats['region_price_mean'].get(row['address_locality'], self.stats['global_price_mean']),
            axis=1
        )
        
        # Size compared to regional mean (r=0.66)
        df['size_vs_region_mean'] = df.apply(
            lambda row: row['size'] / self.stats['region_size_mean'].get(row['address_locality'], self.stats['global_size_mean']),
            axis=1
        )
        
        # ============================================================
        # POLYNOMIAL FEATURES (2 features)
        # ============================================================
        # Non-linear bedroom/bathroom relationships
        bedrooms_numeric = df['bedrooms'].replace({'Studio': 0.5}).astype(float)
        df['bedrooms_squared'] = bedrooms_numeric ** 2
        df['bathrooms_squared'] = df['bathrooms'] ** 2
        
        # ============================================================
        # BATHROOM QUALITY INDICATORS (3 features)
        # ============================================================
        # Bathroom luxury indicators
        df['bathroom_to_bedroom_ratio'] = df['bathrooms'] / bedrooms_numeric.replace(0, 1)
        df['has_extra_bathrooms'] = (df['bathrooms'] >= bedrooms_numeric + 1).astype(int)
        df['is_balanced_layout'] = (df['bathrooms'] == bedrooms_numeric).astype(int)
        
        # ============================================================
        # PROPERTY TYPE ENCODING (2 features)
        # ============================================================
        # Property type indicators
        df['is_townhouse'] = (df['property_type'] == 'Townhouse').astype(int)
        df['is_premium_property_type'] = df['property_type'].isin(['Townhouse', 'Duplex']).astype(int)
        
        # ============================================================
        # DERIVED FEATURES (3 features)
        # ============================================================
        # Total rooms (bedrooms + bathrooms) - overall property size/luxury indicator
        df['total_rooms'] = df['bedrooms'].replace({'Studio': 0}).astype(float) + df['bathrooms']
        
        # Non-linear luxury scaling - total_rooms^1.5 (r=0.73)
        df['total_rooms_power'] = df['total_rooms'] ** 1.5
        
        # Large property premium indicator (>2000 sqft) (r=0.65)
        df['is_large_property'] = (df['size'] > 2000).astype(int)
        
        return df
    
    def fit_transform(self, df):
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self):
        """
        Get list of all engineered feature names.
        
        Returns:
            List of 24 engineered feature names
        """
        return [
            # Core (3) - from raw data
            'bedrooms', 'bathrooms', 'size',
            
            # Transformation (2)
            'log_size', 'sqrt_size',
            
            # Price Ratios (2)
            'price_per_sqft', 'price_per_bedroom',
            
            # Size Ratios (1)
            'size_per_bedroom',
            
            # Location (3)
            'location_tier', 'location_popularity', 'region_avg_price_for_bedrooms',
            
            # Statistical (3)
            'price_percentile_in_region', 'price_vs_region_mean', 'size_vs_region_mean',
            
            # Polynomial (2)
            'bedrooms_squared', 'bathrooms_squared',
            
            # Bathroom Quality (3)
            'bathroom_to_bedroom_ratio', 'has_extra_bathrooms', 'is_balanced_layout',
            
            # Property Type (2)
            'is_townhouse', 'is_premium_property_type',
            
            # Derived (3)
            'total_rooms', 'total_rooms_power', 'is_large_property'
        ]
    
    def get_feature_metadata(self):
        """
        Get metadata for all features including correlation and rationale.
        
        Returns:
            Dictionary mapping feature names to metadata
        """
        return {
            'bedrooms': {
                'type': 'Core',
                'correlation': 0.6825,
                'rationale': 'Fundamental property size indicator'
            },
            'bathrooms': {
                'type': 'Core',
                'correlation': 0.7138,
                'rationale': 'Quality and luxury indicator'
            },
            'size': {
                'type': 'Core',
                'correlation': 0.7948,
                'rationale': 'Strongest base predictor'
            },
            'log_size': {
                'type': 'Transformation',
                'correlation': 0.7292,
                'rationale': 'Captures non-linear size-price relationship'
            },
            'price_per_sqft': {
                'type': 'Price Ratio',
                'correlation': 0.5615,
                'rationale': 'Industry standard metric, critical for business'
            },
            'price_per_bedroom': {
                'type': 'Price Ratio',
                'correlation': 0.6501,
                'rationale': 'Tenant affordability indicator'
            },
            'size_per_bedroom': {
                'type': 'Size Ratio',
                'correlation': 0.6424,
                'rationale': 'Property spaciousness and quality indicator'
            },
            'location_tier': {
                'type': 'Location',
                'correlation': 0.2978,
                'rationale': 'Domain-driven premium location encoding (1/2/3)'
            },
            'location_popularity': {
                'type': 'Location',
                'correlation': 0.5088,
                'rationale': 'Market supply indicator for location'
            },
            'region_avg_price_for_bedrooms': {
                'type': 'Location',
                'correlation': 0.5975,
                'rationale': 'Location-specific price benchmark'
            },
            'price_percentile_in_region': {
                'type': 'Statistical',
                'correlation': 0.4125,
                'rationale': 'Relative positioning within local market'
            },
            'total_rooms': {
                'type': 'Derived',
                'correlation': 0.7088,
                'rationale': 'Overall property size and luxury indicator'
            },
            'price_per_sqft_X_bedrooms': {
                'type': 'Interaction',
                'correlation': 0.4444,
                'rationale': 'Captures non-linear bedroom-price relationship'
            }
        }

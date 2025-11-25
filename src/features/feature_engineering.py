"""
Feature Engineering Pipeline for Abu Dhabi Rental Price Predictor

Creates 38 sophisticated features from 14 base fields to compensate for missing
amenity data due to AWS WAF blocking on PropertyFinder individual pages.

Author: Property Finder ML Team
Date: November 22, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re


class FeatureEngineer:
    """
    Sophisticated feature engineering pipeline to create 38 features from 14 base fields.
    Compensates for missing amenity data with domain-driven feature creation.
    
    Feature Groups:
    1. Price-based features (8): Price per sqft, per bedroom, per bathroom, ratios
    2. Size-based features (7): Size per room, deviations, categories
    3. Location features (6): Tier encoding, premium flags, regional stats
    4. Temporal features (5): Days since listed, freshness indicators
    5. Quality/Badge features (4): Verified, SuperAgent, Spotlight badges
    6. Bedroom/Bathroom features (4): Ratios, studio flag, large unit flag
    7. Interaction features (4): Cross-feature products for complex patterns
    
    Usage:
        engineer = FeatureEngineer()
        df_train_engineered = engineer.fit_transform(df_train)
        df_test_engineered = engineer.transform(df_test)
    """
    
    def __init__(self):
        """Initialize feature engineer with location tier mapping."""
        # Location tier mapping (based on Abu Dhabi market analysis)
        self.location_tiers = {
            'Saadiyat Island': 1,    # Ultra-premium (cultural district, beaches)
            'Al Maryah Island': 1,   # Ultra-premium (financial district, luxury)
            'Yas Island': 2,         # Premium (entertainment, F1 circuit)
            'Al Reem Island': 2,     # Premium (modern towers, waterfront)
            'Corniche Road': 2,      # Premium (beachfront, city center)
            'Masdar City': 3,        # Mid-premium (sustainable city)
            'Al Raha Beach': 3       # Mid-premium (residential, beach access)
        }
        
        # Pre-computed statistics (will be calculated from training data)
        self.stats = {}
    
    def fit(self, df):
        """
        Calculate statistics from training data for feature engineering.
        
        Args:
            df: Training DataFrame with raw features
            
        Returns:
            self: Fitted FeatureEngineer instance
        """
        # Regional statistics (price and size by location)
        self.stats['region_price_mean'] = df.groupby('address_locality')['price'].mean().to_dict()
        self.stats['region_price_std'] = df.groupby('address_locality')['price'].std().to_dict()
        self.stats['region_size_mean'] = df.groupby('address_locality')['size'].mean().to_dict()
        
        # Bedroom statistics (price and size by bedroom count)
        self.stats['bedroom_price_mean'] = df.groupby('bedrooms')['price'].mean().to_dict()
        self.stats['bedroom_size_mean'] = df.groupby('bedrooms')['size'].mean().to_dict()
        
        # Global statistics (for fallback when specific stats unavailable)
        self.stats['global_price_mean'] = df['price'].mean()
        self.stats['global_price_std'] = df['price'].std()
        self.stats['global_size_mean'] = df['size'].mean()
        
        return self
    
    def transform(self, df):
        """
        Transform dataset with 38 engineered features.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with 38 additional engineered features
        """
        df = df.copy()
        
        # ============================================================
        # GROUP 1: PRICE-BASED FEATURES (8 features)
        # ============================================================
        
        # Basic price ratios
        df['price_per_sqft'] = df['price'] / df['size'].replace(0, 1)
        df['price_per_bedroom'] = df['price'] / df['bedrooms'].replace({'Studio': 0.5}).replace(0, 1).astype(float)
        df['price_per_bathroom'] = df['price'] / df['bathrooms'].replace(0, 1)
        df['price_per_room'] = df['price'] / (df['bedrooms'].replace({'Studio': 0}).astype(float) + df['bathrooms'] + 1)
        
        # Price deviation from regional mean (z-score)
        df['price_vs_region_mean'] = df.apply(
            lambda row: (row['price'] - self.stats['region_price_mean'].get(row['address_locality'], self.stats['global_price_mean'])) / 
                       self.stats['region_price_std'].get(row['address_locality'], self.stats['global_price_std']),
            axis=1
        )
        
        # Price percentile within region (0-1 rank)
        df['price_percentile_in_region'] = df.groupby('address_locality')['price'].rank(pct=True)
        
        # Log-transformed price features (for non-linear patterns)
        df['log_price'] = np.log1p(df['price'])
        df['sqrt_price'] = np.sqrt(df['price'])
        
        # ============================================================
        # GROUP 2: SIZE-BASED FEATURES (7 features)
        # ============================================================
        
        # Basic size ratios
        df['size_per_bedroom'] = df['size'] / df['bedrooms'].replace({'Studio': 0.5}).replace(0, 1).astype(float)
        df['size_per_bathroom'] = df['size'] / df['bathrooms'].replace(0, 1)
        df['size_per_room'] = df['size'] / (df['bedrooms'].replace({'Studio': 0}).astype(float) + df['bathrooms'] + 1)
        
        # Size deviation from regional mean (normalized)
        df['size_vs_region_mean'] = df.apply(
            lambda row: (row['size'] - self.stats['region_size_mean'].get(row['address_locality'], self.stats['global_size_mean'])) / 
                       self.stats['global_size_mean'],
            axis=1
        )
        
        # Size category (binned by sqft)
        df['size_category'] = pd.cut(df['size'], bins=[0, 500, 1000, 1500, 2500, 100000], 
                                      labels=['Compact', 'Medium', 'Large', 'Very Large', 'Mansion'])
        
        # Log-transformed size features
        df['log_size'] = np.log1p(df['size'])
        df['sqrt_size'] = np.sqrt(df['size'])
        
        # ============================================================
        # GROUP 3: LOCATION FEATURES (6 features)
        # ============================================================
        
        # Location tier (1=Ultra-premium, 2=Premium, 3=Mid-premium)
        df['location_tier'] = df['address_locality'].map(self.location_tiers)
        
        # Binary flags for each tier
        df['is_ultra_premium'] = (df['location_tier'] == 1).astype(int)
        df['is_premium'] = (df['location_tier'] == 2).astype(int)
        df['is_mid_premium'] = (df['location_tier'] == 3).astype(int)
        
        # Regional average price for this location
        df['region_avg_price_for_bedrooms'] = df.apply(
            lambda row: self.stats['region_price_mean'].get(row['address_locality'], self.stats['global_price_mean']),
            axis=1
        )
        
        # Location popularity (count of properties in this location)
        df['location_popularity'] = df.groupby('address_locality')['property_id'].transform('count')
        
        # ============================================================
        # GROUP 4: TEMPORAL FEATURES (5 features)
        # ============================================================
        
        # Parse listed_date: "Listed X days/weeks/months ago"
        def parse_listed_date(text):
            """Extract number of days from listing date text."""
            if pd.isna(text) or text == '':
                return np.nan
            match = re.search(r'(\d+)\s+(day|week|month)', str(text).lower())
            if match:
                num = int(match.group(1))
                unit = match.group(2)
                if unit == 'day':
                    return num
                elif unit == 'week':
                    return num * 7
                elif unit == 'month':
                    return num * 30
            return np.nan
        
        df['days_since_listed'] = df['listed_date'].apply(parse_listed_date)
        df['days_since_listed'].fillna(df['days_since_listed'].median(), inplace=True)
        
        # Freshness indicators
        df['is_fresh_listing'] = (df['days_since_listed'] <= 7).astype(int)
        df['is_old_listing'] = (df['days_since_listed'] >= 60).astype(int)
        
        # Listing age category
        df['listing_age_category'] = pd.cut(df['days_since_listed'], bins=[0, 7, 30, 90, 365, 10000], 
                                              labels=['Fresh', 'Recent', 'Moderate', 'Old', 'Stale'])
        
        # Log-transformed days (reduces skewness)
        df['log_days_since_listed'] = np.log1p(df['days_since_listed'])
        
        # ============================================================
        # GROUP 5: QUALITY/BADGE FEATURES (4 features)
        # ============================================================
        
        # Badge indicators (Verified, SuperAgent, Spotlight)
        df['has_verified_badge'] = df['badges'].str.contains('Verified', case=False, na=False).astype(int)
        df['has_superagent_badge'] = df['badges'].str.contains('SuperAgent', case=False, na=False).astype(int)
        df['has_spotlight_badge'] = df['badges'].str.contains('Spotlight', case=False, na=False).astype(int)
        
        # Total badge count
        df['badge_count'] = df['badges'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0)
        
        # ============================================================
        # GROUP 6: BEDROOM/BATHROOM FEATURES (4 features)
        # ============================================================
        
        # Bedroom to bathroom ratio
        df['bedroom_bathroom_ratio'] = df['bedrooms'].replace({'Studio': 0}).astype(float) / df['bathrooms'].replace(0, 1)
        
        # Property type indicators
        df['is_studio'] = (df['bedrooms'] == 'Studio').astype(int)
        df['is_large_unit'] = (df['bedrooms'].replace({'Studio': 0}).astype(float) >= 3).astype(int)
        
        # Total rooms (bedrooms + bathrooms)
        df['total_rooms'] = df['bedrooms'].replace({'Studio': 0}).astype(float) + df['bathrooms']
        
        # ============================================================
        # GROUP 7: INTERACTION FEATURES (4 features)
        # ============================================================
        
        # Cross-feature interactions for complex patterns
        df['price_per_sqft_X_tier'] = df['price_per_sqft'] * df['location_tier']
        df['size_X_tier'] = df['size'] * df['location_tier']
        df['bedrooms_X_tier'] = df['bedrooms'].replace({'Studio': 0}).astype(float) * df['location_tier']
        df['price_per_sqft_X_bedrooms'] = df['price_per_sqft'] * df['bedrooms'].replace({'Studio': 0.5}).astype(float)
        
        return df
    
    def fit_transform(self, df):
        """
        Fit and transform in one step.
        
        Args:
            df: Training DataFrame with raw features
            
        Returns:
            DataFrame with 38 additional engineered features
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self):
        """
        Get list of all engineered feature names.
        
        Returns:
            List of 38 engineered feature names
        """
        return [
            # Price-based (8)
            'price_per_sqft', 'price_per_bedroom', 'price_per_bathroom', 'price_per_room',
            'price_vs_region_mean', 'price_percentile_in_region', 'log_price', 'sqrt_price',
            
            # Size-based (7)
            'size_per_bedroom', 'size_per_bathroom', 'size_per_room', 'size_vs_region_mean',
            'size_category', 'log_size', 'sqrt_size',
            
            # Location (6)
            'location_tier', 'is_ultra_premium', 'is_premium', 'is_mid_premium',
            'region_avg_price_for_bedrooms', 'location_popularity',
            
            # Temporal (5)
            'days_since_listed', 'is_fresh_listing', 'is_old_listing',
            'listing_age_category', 'log_days_since_listed',
            
            # Quality/Badges (4)
            'has_verified_badge', 'has_superagent_badge', 'has_spotlight_badge', 'badge_count',
            
            # Bedroom/Bathroom (4)
            'bedroom_bathroom_ratio', 'is_studio', 'is_large_unit', 'total_rooms',
            
            # Interactions (4)
            'price_per_sqft_X_tier', 'size_X_tier', 'bedrooms_X_tier', 'price_per_sqft_X_bedrooms'
        ]

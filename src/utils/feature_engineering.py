"""
Feature Engineering Utilities
Reusable functions for creating model features
"""
import numpy as np
import pandas as pd


def create_log_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Create log-transformed features

    Args:
        df: DataFrame
        columns: List of columns to log-transform

    Returns:
        DataFrame with log features added
    """
    df = df.copy()
    for col in columns:
        df[f'log_{col}'] = np.log1p(df[col])
    return df


def create_interaction_features(df: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """
    Create interaction features between column pairs

    Args:
        df: DataFrame
        pairs: List of tuples [(col1, col2), ...]

    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    for col1, col2 in pairs:
        df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
    return df


def create_amenity_features(df: pd.DataFrame, amenity_columns: list) -> pd.DataFrame:
    """
    Convert amenity columns to binary features

    Args:
        df: DataFrame
        amenity_columns: List of amenity column names

    Returns:
        DataFrame with binary amenity features
    """
    df = df.copy()
    amenity_count = pd.Series([0] * len(df), index=df.index)
    
    for col in amenity_columns:
        if col in df.columns:
            feature_name = col.replace('amenity_', 'has_').lower()
            df[feature_name] = (df[col] > 0).astype(int)
            amenity_count += df[feature_name]
        else:
            feature_name = col.replace('amenity_', 'has_').lower()
            df[feature_name] = 0
    
    df['amenity_count'] = amenity_count
    return df


def create_region_features(df: pd.DataFrame, region_col: str = 'address_region') -> pd.DataFrame:
    """
    Create region-based statistical features

    Args:
        df: DataFrame
        region_col: Name of region column

    Returns:
        DataFrame with region features added
    """
    df = df.copy()

    # Region price statistics
    region_stats = df.groupby(region_col).agg({
        'price': ['mean', 'median', 'std']
    }).reset_index()
    region_stats.columns = [region_col, 'region_price_mean', 'region_price_median', 'region_price_std']

    df = df.merge(region_stats, on=region_col, how='left')

    # Fill missing with overall statistics
    df['region_price_mean'].fillna(df['price'].mean(), inplace=True)
    df['region_price_median'].fillna(df['price'].median(), inplace=True)
    df['region_price_std'].fillna(df['price'].std(), inplace=True)

    return df


def create_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create property-specific features

    Args:
        df: DataFrame with bedrooms, bathrooms, size

    Returns:
        DataFrame with property features added
    """
    df = df.copy()

    # Total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # Price per sqft (if price exists)
    if 'price' in df.columns and 'size' in df.columns:
        df['price_per_sqft'] = df['price'] / df['size']

    # Bedroom size interaction
    if 'bedrooms' in df.columns and 'size' in df.columns:
        df['bed_size_interaction'] = df['bedrooms'] * np.log1p(df['size'])

    return df


def build_v62_features(
    bedrooms: int,
    bathrooms: int,
    size: float,
    location: str,
    region_means: dict,
    has_water_view: bool = False,
    has_parking: bool = False,
    has_pool: bool = False,
    has_gym: bool = False,
    has_balcony: bool = False
) -> dict:
    """
    Build feature dictionary for v6.2 production model

    Args:
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        size: Size in sqft
        location: Region/area name
        region_means: Dict mapping regions to average prices
        has_water_view: Has water view
        has_parking: Has parking
        has_pool: Has pool
        has_gym: Has gym
        has_balcony: Has balcony

    Returns:
        Dictionary of 13 features for v6.2 model
    """
    # Get region price mean
    region_price_mean = region_means.get(location, 100000)

    # Calculate derived features
    log_size = np.log1p(size)
    total_rooms = bedrooms + bathrooms
    price_per_sqft_est = region_price_mean / size
    bed_size_interaction = bedrooms * log_size

    # Build feature dict
    features = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'size': size,
        'region_price_mean': region_price_mean,
        'log_size': log_size,
        'total_rooms': total_rooms,
        'price_per_sqft_est': price_per_sqft_est,
        'has_water_view': int(has_water_view),
        'has_parking': int(has_parking),
        'has_pool': int(has_pool),
        'has_gym': int(has_gym),
        'has_balcony': int(has_balcony),
        'bed_size_interaction': bed_size_interaction
    }

    return features

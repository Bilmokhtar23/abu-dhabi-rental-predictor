"""
Feature Processing Module
Handles feature creation and preprocessing for predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def preprocess_location(location: str) -> str:
    """
    Preprocess location string for consistency

    Args:
        location: Raw location string

    Returns:
        Processed location string
    """
    if not location:
        return "Unknown"

    # Standardize common locations
    location = location.strip().title()

    # Handle common variations
    location_map = {
        "Al Reem Island": "Al Reem Island",
        "Yas Island": "Yas Island",
        "Saadiyat Island": "Saadiyat Island",
        "Corniche": "Corniche",
        "Al Maryah Island": "Al Maryah Island",
        # Add more mappings as needed
    }

    return location_map.get(location, location)


def create_prediction_features(
    bedrooms: int,
    bathrooms: int,
    size: float,
    location: str,
    has_water_view: bool = False,
    has_gym: bool = False,
    has_parking: bool = False
) -> Dict[str, Any]:
    """
    Create feature dictionary for prediction

    Args:
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        size: Property size in sqft
        location: Property location
        has_water_view: Water view amenity
        has_gym: Gym amenity
        has_parking: Parking amenity

    Returns:
        Feature dictionary
    """
    # Preprocess inputs
    location = preprocess_location(location)

    # Create features
    features = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'size': size,
        'location': location,
        'has_water_view': int(has_water_view),
        'has_gym': int(has_gym),
        'has_parking': int(has_parking),
        # Derived features
        'bedrooms_per_bathroom': bedrooms / max(bathrooms, 1),
        'size_per_bedroom': size / max(bedrooms, 1),
        'log_size': np.log1p(size),
        'location_encoded': hash(location) % 1000  # Simple encoding
    }

    return features


def validate_features(features: Dict[str, Any]) -> bool:
    """
    Validate feature values

    Args:
        features: Feature dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check ranges - reasonable limits for Abu Dhabi apartments
        assert 0 <= features['bedrooms'] <= 5  # Max 5 bedrooms for apartments
        assert 0 <= features['bathrooms'] <= 5  # Max 5 bathrooms
        assert 100 <= features['size'] <= 10000
        assert isinstance(features['location'], str)
        return True
    except (AssertionError, KeyError, TypeError):
        return False
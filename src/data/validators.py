"""
Data validation using Pydantic
Ensures data quality before training or prediction
"""
from pydantic import BaseModel, field_validator, Field, model_validator
from typing import Optional, Literal
import pandas as pd


class PropertyData(BaseModel):
    """Validation schema for property data"""

    # Required fields
    price: float = Field(..., gt=10000, lt=1000000, description="Annual rent in AED")
    size: float = Field(..., gt=100, lt=10000, description="Size in square feet")
    bedrooms: int = Field(..., ge=0, le=10, description="Number of bedrooms (0 = Studio)")
    bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    address_region: str = Field(..., min_length=2, description="Property region/area")

    # Optional fields
    latitude: Optional[float] = Field(None, ge=22.0, le=26.0)
    longitude: Optional[float] = Field(None, ge=52.0, le=56.0)
    floor_size: Optional[float] = None

    # Amenities
    amenity_View_of_Water: Optional[Literal["Yes", "No"]] = "No"
    amenity_Covered_Parking: Optional[Literal["Yes", "No"]] = "No"
    amenity_Shared_Pool: Optional[Literal["Yes", "No"]] = "No"
    amenity_Shared_Gym: Optional[Literal["Yes", "No"]] = "No"
    amenity_Balcony: Optional[Literal["Yes", "No"]] = "No"

    class Config:
        extra = "allow"  # Allow additional fields

    @field_validator('price')
    @classmethod
    def validate_price_range(cls, v):
        """Ensure price is reasonable for Abu Dhabi market"""
        if not (25000 <= v <= 450000):
            raise ValueError(f'Price {v} AED outside expected range (25K-450K AED)')
        return v

    @field_validator('size')
    @classmethod
    def validate_size_range(cls, v):
        """Ensure size is reasonable"""
        if not (250 <= v <= 5000):
            raise ValueError(f'Size {v} sqft outside expected range (250-5000 sqft)')
        return v

    @model_validator(mode='after')
    def validate_bathrooms_and_psf(self):
        """Validate bathrooms ratio and price per sqft"""
        # Check bathrooms
        if self.bathrooms > self.bedrooms + 3:
            raise ValueError(f'Bathrooms ({self.bathrooms}) too many for {self.bedrooms} bedrooms')

        # Check price per sqft
        psf = self.price / self.size
        if not (40 <= psf <= 300):
            raise ValueError(f'Price/sqft {psf:.2f} AED outside range (40-300 AED/sqft)')

        return self

    @property
    def price_per_sqft(self) -> float:
        """Calculate price per square foot"""
        return self.price / self.size


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Validate entire DataFrame

    Returns:
        (valid_df, errors): Valid rows and list of errors
    """
    valid_rows = []
    errors = []

    for idx, row in df.iterrows():
        try:
            # Attempt to validate
            PropertyData(**row.to_dict())
            valid_rows.append(row)
        except Exception as e:
            errors.append({
                'row': idx,
                'error': str(e),
                'data': row.to_dict()
            })

    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()

    return valid_df, errors


def validate_single_property(
    price: float,
    size: float,
    bedrooms: int,
    bathrooms: int,
    address_region: str,
    **kwargs
) -> dict:
    """
    Validate a single property for prediction

    Returns:
        dict with 'valid' (bool) and 'errors' (list)
    """
    try:
        PropertyData(
            price=price,
            size=size,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            address_region=address_region,
            **kwargs
        )
        return {'valid': True, 'errors': []}
    except Exception as e:
        return {'valid': False, 'errors': [str(e)]}


# Example usage
if __name__ == "__main__":
    # Valid property
    valid_prop = PropertyData(
        price=120000,
        size=1200,
        bedrooms=2,
        bathrooms=2,
        address_region="Al Reem Island"
    )
    print(f"✓ Valid property: {valid_prop.price} AED for {valid_prop.size} sqft")
    print(f"  Price/sqft: {valid_prop.price_per_sqft:.2f} AED")

    # Invalid property (will raise error)
    try:
        invalid_prop = PropertyData(
            price=10000000,  # Too high
            size=1200,
            bedrooms=2,
            bathrooms=2,
            address_region="Al Reem Island"
        )
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")

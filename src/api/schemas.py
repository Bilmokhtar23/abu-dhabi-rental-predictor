"""
API Schemas and Models
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class PredictionRequest(BaseModel):
    """Request model for price prediction"""
    bedrooms: int = Field(..., ge=0, le=10, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, le=10, description="Number of bathrooms")
    size: float = Field(..., gt=0, le=10000, description="Property size in sqft")
    location: str = Field(..., min_length=1, description="Property location")
    has_water_view: bool = Field(False, description="Water view amenity")
    has_gym: bool = Field(False, description="Gym amenity")
    has_parking: bool = Field(False, description="Parking amenity")


class PredictionResponse(BaseModel):
    """Response model for price prediction"""
    predicted_price: float = Field(..., description="Predicted rental price in AED")
    confidence_score: Optional[float] = Field(None, description="Model confidence score")
    features_used: List[str] = Field(..., description="Features used in prediction")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Model version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    properties: List[PredictionRequest] = Field(..., description="List of properties to predict")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
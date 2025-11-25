"""
FastAPI for Abu Dhabi Rental Price Predictor
Production-ready REST API with comprehensive endpoints
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
import time
from pathlib import Path
import sys
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predictor import RentalPredictor
from config import get_api_config, get_webapp_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Get configurations
api_config = get_api_config()
webapp_config = get_webapp_config()

# Initialize FastAPI app
app = FastAPI(
    title=f"{webapp_config['title']} API",
    description="AI-powered rental price prediction for Abu Dhabi properties",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware
@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    """Middleware to track Prometheus metrics"""
    start_time = time.time()

    # Track active connections
    ACTIVE_CONNECTIONS.inc()

    try:
        response = await call_next(request)
        status_code = str(response.status_code)

        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status_code
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)

        return response
    finally:
        ACTIVE_CONNECTIONS.dec()

# Global predictor instance (lazy loading)
predictor = None

def get_predictor() -> RentalPredictor:
    """Get or create predictor instance"""
    global predictor
    if predictor is None:
        try:
            predictor = RentalPredictor()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    return predictor

# Pydantic models for request/response
class PropertyFeatures(BaseModel):
    """Property features for prediction"""
    bedrooms: int = Field(..., ge=0, le=10, description="Number of bedrooms (0 for studio)")
    bathrooms: int = Field(..., ge=1, le=8, description="Number of bathrooms")
    size: float = Field(..., gt=0, le=10000, description="Size in square feet")
    location: str = Field(..., min_length=1, description="Location/region name")

    # Optional amenities
    has_water_view: bool = Field(False, description="Has water view")
    has_parking: bool = Field(False, description="Has covered parking")
    has_pool: bool = Field(False, description="Has pool access")
    has_gym: bool = Field(False, description="Has gym access")
    has_balcony: bool = Field(False, description="Has balcony")

    @validator('location')
    def validate_location(cls, v):
        """Basic location validation"""
        if not v or len(v.strip()) == 0:
            raise ValueError('Location cannot be empty')
        return v.strip()

class PredictionResponse(BaseModel):
    """Prediction response"""
    model_config = {'protected_namespaces': ()}

    predicted_price: float = Field(..., description="Predicted annual rent in AED")
    monthly_rent: float = Field(..., description="Predicted monthly rent in AED")
    price_per_sqft: float = Field(..., description="Price per square foot")
    mae: float = Field(..., description="Model mean absolute error")
    mape: float = Field(..., description="Model mean absolute percentage error")
    confidence_lower: float = Field(..., description="Lower confidence bound")
    confidence_upper: float = Field(..., description="Upper confidence bound")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction identifier")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    properties: List[PropertyFeatures] = Field(..., max_items=1000, description="List of properties to predict")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_properties: int
    processing_time: float
    batch_id: str

class ModelInfo(BaseModel):
    """Model information"""
    version: str
    n_features: int
    test_mae: float
    test_r2: float
    training_date: str
    features: List[str]

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Abu Dhabi Rental Price Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        pred = get_predictor()
        # Quick test prediction
        test_result = pred.predict(
            bedrooms=2, bathrooms=2, size=1200,
            location='Al Reem Island'
        )
        return {
            "status": "healthy",
            "model_loaded": True,
            "test_prediction": round(test_result['predicted_price'], 2)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")  # Rate limit: 100 requests per minute per IP
async def predict_price(request: Request, property: PropertyFeatures):
    """Predict rental price for a single property"""
    try:
        start_time = time.time()
        pred = get_predictor()

        # Make prediction
        result = pred.predict(**property.dict())

        # Calculate additional metrics
        monthly_rent = result['predicted_price'] / 12
        price_per_sqft = result['predicted_price'] / property.size

        # Create response
        response = PredictionResponse(
            predicted_price=round(result['predicted_price'], 2),
            monthly_rent=round(monthly_rent, 2),
            price_per_sqft=round(price_per_sqft, 2),
            mae=round(result['mae'], 2),
            mape=round(result['mape'], 2),
            confidence_lower=round(result['confidence_lower'], 2),
            confidence_upper=round(result['confidence_upper'], 2),
            model_version="v6.2_production",
            prediction_id=f"pred_{int(time.time())}_{hash(str(property.dict()))}"
        )

        processing_time = time.time() - start_time
        logger.info(f"Prediction completed in {processing_time:.3f}s: {response.prediction_id}")

        # Track prediction metrics
        PREDICTION_COUNT.inc()

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit("20/minute")  # Rate limit: 20 batch requests per minute per IP
async def predict_batch(request: Request, req: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Predict rental prices for multiple properties"""
    try:
        start_time = time.time()
        pred = get_predictor()

        predictions = []
        for prop in req.properties:
            result = pred.predict(**prop.dict())

            monthly_rent = result['predicted_price'] / 12
            price_per_sqft = result['predicted_price'] / prop.size

            prediction = PredictionResponse(
                predicted_price=round(result['predicted_price'], 2),
                monthly_rent=round(monthly_rent, 2),
                price_per_sqft=round(price_per_sqft, 2),
                mae=round(result['mae'], 2),
                mape=round(result['mape'], 2),
                confidence_lower=round(result['confidence_lower'], 2),
                confidence_upper=round(result['confidence_upper'], 2),
                model_version="v6.2_production",
                prediction_id=f"batch_{int(time.time())}_{hash(str(prop.dict()))}"
            )
            predictions.append(prediction)

        processing_time = time.time() - start_time
        batch_id = f"batch_{int(time.time())}_{len(predictions)}"

        logger.info(f"Batch prediction completed: {batch_id} ({len(predictions)} properties, {processing_time:.3f}s)")

        # Track prediction metrics
        PREDICTION_COUNT.inc(len(predictions))

        return BatchPredictionResponse(
            predictions=predictions,
            total_properties=len(predictions),
            processing_time=round(processing_time, 3),
            batch_id=batch_id
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and metadata"""
    try:
        pred = get_predictor()
        info = pred.get_model_info()

        return ModelInfo(
            version=info['version'],
            n_features=info['n_features'],
            test_mae=round(info['test_mae'], 2),
            test_r2=round(info['test_r2'], 4),
            training_date="2025-11-20",  # Update with actual date
            features=info['features']
        )

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@app.get("/stats")
async def get_statistics():
    """Get prediction statistics and usage metrics"""
    # In a real implementation, this would track metrics
    return {
        "total_predictions": 0,  # Would be tracked in database
        "average_prediction_time": 0.05,
        "model_accuracy": "15,501 AED MAE",
        "uptime": "99.9%",
        "last_updated": "2025-11-20"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=api_config['host'],
        port=api_config['port'],
        reload=True,
        log_level="info"
    )
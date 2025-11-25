"""
Model Prediction Utilities
Handles loading models and making predictions
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from config import get_model_config, get_data_config


class RentalPredictor:
    """
    Production rental price predictor for v6.2 model
    """

    def __init__(self, model_dir: str = None):
        """
        Initialize predictor by loading model artifacts

        Args:
            model_dir: Path to model artifacts directory (optional, uses config)
        """
        if model_dir is None:
            model_config = get_model_config()
            model_dir = f"model_outputs/{model_config['version']}/models"

        self.model_dir = Path(model_dir)
        self.stacker = None
        self.scaler = None
        self.features = None
        self.region_means = None
        self.metadata = None
        self.load_model()

    def load_model(self) -> None:
        """Load all model artifacts"""
        try:
            self.stacker = joblib.load(self.model_dir / 'stacker.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            self.features = joblib.load(self.model_dir / 'features.pkl')
            self.region_means = joblib.load(self.model_dir / 'region_means.pkl')
            self.metadata = joblib.load(self.model_dir / 'metadata.pkl')
            print(f"✓ Loaded v6.2 model from {self.model_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def create_features(
        self,
        bedrooms: int,
        bathrooms: int,
        size: float,
        location: str,
        has_water_view: bool = False,
        has_parking: bool = False,
        has_pool: bool = False,
        has_gym: bool = False,
        has_balcony: bool = False
    ) -> pd.DataFrame:
        """
        Create feature DataFrame for prediction

        Args:
            bedrooms: Number of bedrooms
            bathrooms: Number of bathrooms
            size: Size in sqft
            location: Region/area
            has_water_view: Has water view
            has_parking: Has parking
            has_pool: Has pool
            has_gym: Has gym
            has_balcony: Has balcony

        Returns:
            DataFrame with features in correct order
        """
        # Get region statistics
        region_price_mean = self.region_means.get(location, 100000)

        # Calculate derived features
        log_size = np.log1p(size)
        total_rooms = bedrooms + bathrooms
        price_per_sqft_est = region_price_mean / size
        bed_size_interaction = bedrooms * log_size

        # Build feature dict
        feature_dict = {
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

        # Create DataFrame in correct feature order
        return pd.DataFrame([feature_dict])[self.features]

    def predict(
        self,
        bedrooms: int,
        bathrooms: int,
        size: float,
        location: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict rental price

        Args:
            bedrooms: Number of bedrooms
            bathrooms: Number of bathrooms
            size: Size in sqft
            location: Region/area
            **kwargs: Amenity flags

        Returns:
            Dict with prediction and confidence interval
        """
        # Create features
        X = self.create_features(
            bedrooms, bathrooms, size, location, **kwargs
        )

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict (model outputs log-transformed price)
        y_pred_log = self.stacker.predict(X_scaled)[0]

        # Inverse transform
        predicted_price = np.expm1(y_pred_log)

        # Get model error for confidence interval
        mae = self.metadata.get('test_mae', 15501)
        mape = self.metadata.get('test_mape', 10.7)

        return {
            'predicted_price': predicted_price,
            'mae': mae,
            'mape': mape,
            'confidence_lower': predicted_price - mae,
            'confidence_upper': predicted_price + mae
        }

    def predict_batch(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict prices for multiple properties

        Args:
            properties_df: DataFrame with columns matching create_features args

        Returns:
            DataFrame with predictions added
        """
        predictions = []

        for _, row in properties_df.iterrows():
            result = self.predict(
                bedrooms=row['bedrooms'],
                bathrooms=row['bathrooms'],
                size=row['size'],
                location=row.get('location', row.get('address_region')),
                has_water_view=row.get('has_water_view', False),
                has_parking=row.get('has_parking', False),
                has_pool=row.get('has_pool', False),
                has_gym=row.get('has_gym', False),
                has_balcony=row.get('has_balcony', False)
            )
            predictions.append(result['predicted_price'])

        properties_df['predicted_price'] = predictions
        return properties_df

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            'version': 'v6.2_production',
            'features': self.features,
            'n_features': len(self.features),
            'test_mae': self.metadata.get('test_mae'),
            'test_r2': self.metadata.get('test_r2'),
            'train_mae': self.metadata.get('train_mae'),
            'generalization_ratio': self.metadata.get('generalization_ratio')
        }


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = RentalPredictor()

    # Single prediction
    result = predictor.predict(
        bedrooms=2,
        bathrooms=2,
        size=1200,
        location="Al Reem Island",
        has_parking=True,
        has_pool=True,
        has_gym=True
    )

    print(f"\nPredicted Price: {result['predicted_price']:,.0f} AED")
    print(f"Expected Error: ±{result['mae']:,.0f} AED ({result['mape']:.1f}%)")
    print(f"Confidence: {result['confidence_lower']:,.0f} - {result['confidence_upper']:,.0f} AED")

    # Model info
    info = predictor.get_model_info()
    print(f"\nModel: {info['version']}")
    print(f"Features: {info['n_features']}")
    print(f"Test MAE: {info['test_mae']:,.0f} AED")
    print(f"Test R²: {info['test_r2']:.4f}")

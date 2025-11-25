"""
Data Loading and Preprocessing
Utilities for loading and cleaning rental property data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from config import get_data_config


def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """
    Load raw property data from CSV

    Args:
        file_path: Path to CSV file (optional, uses config)

    Returns:
        DataFrame with raw data
    """
    if file_path is None:
        data_config = get_data_config()
        file_path = data_config['raw_path']

    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} properties from {file_path}")
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and filter property data

    Args:
        df: Raw DataFrame
        verbose: Print cleaning statistics

    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)

    # Remove rows with missing critical fields
    df = df.dropna(subset=['price', 'size', 'bedrooms', 'bathrooms', 'address_region'])

    # Convert to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

    # Handle Studio apartments
    df['bedrooms'] = df['bedrooms'].replace('Studio', 0)
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')

    # Remove remaining NaN values
    df = df.dropna(subset=['price', 'size', 'bedrooms', 'bathrooms'])

    # Calculate price per sqft
    df['price_per_sqft'] = df['price'] / df['size']

    # Filter outliers
    df = df[df['price'].between(25000, 450000)]
    df = df[df['size'].between(250, 5000)]
    df = df[df['price_per_sqft'].between(40, 300)]
    df = df[df['bedrooms'] <= 6]
    df = df[df['bathrooms'] <= 8]

    if verbose:
        print(f"After cleaning: {len(df)} ({len(df)/initial_count*100:.1f}% retained)")
        print(f"Removed: {initial_count - len(df)} properties")

    return df


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = 'price',
    log_transform: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training

    Args:
        df: Cleaned DataFrame
        target_col: Target column name
        log_transform: Apply log transform to target

    Returns:
        (features_df, target_series)
    """
    # Create copy
    df = df.copy()

    # Extract target
    y = df[target_col]

    # Log transform target if requested
    if log_transform:
        y = np.log1p(y)

    # Remove target from features
    X = df.drop(columns=[target_col], errors='ignore')

    return X, y


def save_processed_data(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'parquet'
) -> None:
    """
    Save processed data

    Args:
        df: DataFrame to save
        output_path: Output file path
        format: File format ('parquet' or 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Saved {len(df)} rows to {output_path}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the data

    Args:
        df: DataFrame

    Returns:
        Dictionary with summary stats
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'price_stats': {
            'mean': df['price'].mean(),
            'median': df['price'].median(),
            'min': df['price'].min(),
            'max': df['price'].max(),
            'std': df['price'].std()
        },
        'size_stats': {
            'mean': df['size'].mean(),
            'median': df['size'].median(),
            'min': df['size'].min(),
            'max': df['size'].max()
        },
        'bedroom_distribution': df['bedrooms'].value_counts().to_dict(),
        'n_regions': df['address_region'].nunique(),
        'top_regions': df['address_region'].value_counts().head(5).to_dict()
    }

    return summary


# Example usage
if __name__ == "__main__":
    # Load data
    df = load_raw_data()

    # Clean data
    df_clean = clean_data(df, verbose=True)

    # Get summary
    summary = get_data_summary(df_clean)
    print("\nData Summary:")
    print(f"  Rows: {summary['n_rows']}")
    print(f"  Price range: {summary['price_stats']['min']:,.0f} - {summary['price_stats']['max']:,.0f} AED")
    print(f"  Avg price: {summary['price_stats']['mean']:,.0f} AED")
    print(f"  Regions: {summary['n_regions']}")

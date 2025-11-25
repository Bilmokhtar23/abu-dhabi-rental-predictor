"""
Configuration Management System
Loads configuration from YAML with environment variable support
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container"""
    model: Dict[str, Any]
    data: Dict[str, Any]
    api: Dict[str, Any]
    webapp: Dict[str, Any]
    monitoring: Dict[str, Any]
    database: Dict[str, Any]
    security: Dict[str, Any]
    features: Dict[str, Any]
    training: Dict[str, Any]


def load_config(config_file: str = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides

    Args:
        config_file: Path to config file (default: config/default.yaml)

    Returns:
        Config object with all settings
    """
    if config_file is None:
        config_file = Path(__file__).parent / 'default.yaml'

    # Load YAML configuration
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")

    # Apply environment variable overrides
    config_data = _apply_env_overrides(config_data)

    # Validate configuration
    _validate_config(config_data)

    # Convert to Config object
    return Config(**config_data)


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration

    Supports nested keys with double underscore separation
    Example: MODEL__VERSION=v6.3 overrides config['model']['version']
    """
    for key, value in os.environ.items():
        if key.startswith(('MODEL__', 'DATA__', 'API__', 'WEBAPP__', 'MONITORING__',
                          'DATABASE__', 'SECURITY__', 'FEATURES__', 'TRAINING__')):
            section, setting = key.lower().split('__', 1)
            if section in config and isinstance(config[section], dict):
                # Convert string values to appropriate types
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'

                config[section][setting] = value

    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Required sections
    required_sections = ['model', 'data', 'api', 'webapp']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Model validation
    model = config['model']
    if 'version' not in model:
        raise ValueError("Model version must be specified")
    if 'n_features' in model and not isinstance(model['n_features'], int):
        raise ValueError("Model n_features must be an integer")

    # Data validation
    data = config['data']
    if 'raw_path' not in data:
        raise ValueError("Data raw_path must be specified")
    if 'min_price' in data and 'max_price' in data:
        if data['min_price'] >= data['max_price']:
            raise ValueError("Data min_price must be less than max_price")

    # API validation
    api = config['api']
    if 'port' in api and not (1 <= api['port'] <= 65535):
        raise ValueError("API port must be between 1 and 65535")
    if 'workers' in api and api['workers'] < 1:
        raise ValueError("API workers must be at least 1")

    # Webapp validation
    webapp = config['webapp']
    if 'title' not in webapp:
        raise ValueError("Webapp title must be specified")


# Load configuration on import
config = load_config()


# Convenience access to config sections
def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    return config.model


def get_data_config() -> Dict[str, Any]:
    """Get data configuration"""
    return config.data


def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    return config.api


def get_webapp_config() -> Dict[str, Any]:
    """Get webapp configuration"""
    return config.webapp


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration"""
    return config.monitoring


def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return config.database


def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    return config.security


def get_features_config() -> Dict[str, Any]:
    """Get features configuration"""
    return config.features


def get_training_config() -> Dict[str, Any]:
    """Get training configuration"""
    return config.training

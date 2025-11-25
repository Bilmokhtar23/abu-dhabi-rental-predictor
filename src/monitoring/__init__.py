"""
Monitoring and Logging Utilities
Handles observability and logging for the application
"""
import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """
    Set up logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def log_prediction_request(features: dict, prediction: float):
    """
    Log prediction requests for monitoring

    Args:
        features: Input features
        prediction: Model prediction
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Prediction request: {features} -> {prediction:.2f}")


def log_error(error: Exception, context: str = ""):
    """
    Log errors with context

    Args:
        error: Exception object
        context: Additional context information
    """
    logger = logging.getLogger(__name__)
    logger.error(f"Error in {context}: {str(error)}")
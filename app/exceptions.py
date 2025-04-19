"""
Custom exceptions for property valuation system.
"""

class PropertyValuationError(Exception):
    """Base exception for property valuation errors."""
    pass


class InvalidPropertyTypeError(PropertyValuationError):
    """Raised when property type is invalid or missing."""
    pass


class InvalidLocationError(PropertyValuationError):
    """Raised when location data is invalid or missing."""
    pass


class ModelLoadError(PropertyValuationError):
    """Raised when model loading fails."""
    pass


class FeatureEngineeringError(PropertyValuationError):
    """Raised when feature engineering fails."""
    pass


class DataAccessError(PropertyValuationError):
    """Raised when database access fails."""
    pass


class PredictionError(PropertyValuationError):
    """Raised when prediction process fails."""
    pass
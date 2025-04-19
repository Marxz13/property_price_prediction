"""
Service for loading models and data.
"""
from typing import Dict, List, Tuple, Any
from functools import lru_cache
from flask import current_app
from pymongo import MongoClient

from app.models.data_loader import DataLoader
from app.exceptions import ModelLoadError, DataAccessError


@lru_cache(maxsize=1)
def get_db_client() -> MongoClient:
    """
    Get MongoDB client with caching.

    Returns:
        MongoClient: MongoDB client instance

    Raises:
        DataAccessError: If database connection fails
    """
    try:
        mongodb_uri = current_app.config.get("MONGODB_URI")
        return MongoClient(mongodb_uri)
    except Exception as e:
        current_app.logger.error(f"Failed to connect to MongoDB: {e}")
        raise DataAccessError(f"Failed to connect to MongoDB: {e}")


@lru_cache(maxsize=1)
def load_models_and_data() -> Tuple[Dict[str, Any], Tuple[List[Tuple[float, float]],
List[Tuple[float, float]],
List[Tuple[float, float]]]]:
    """
    Load machine learning models and POI data with caching.

    Returns:
        Tuple containing:
            Dict[str, Any]: Dictionary mapping property types to models
            Tuple: Tuple of landmark, station, and mall coordinates

    Raises:
        ModelLoadError: If model loading fails
        DataAccessError: If data loading fails
    """
    try:
        model_dir = current_app.config.get("MODEL_DIR")
        data_loader = DataLoader(model_dir=model_dir)

        # Load models
        models = data_loader.load_all_models()

        # Load POI data
        poi_data = data_loader.load_poi_data()

        return models, poi_data

    except ModelLoadError as e:
        current_app.logger.error(f"Failed to load models: {e}")
        raise ModelLoadError(f"Failed to load models: {e}")
    except DataAccessError as e:
        current_app.logger.error(f"Failed to load POI data: {e}")
        raise DataAccessError(f"Failed to load POI data: {e}")
    except Exception as e:
        current_app.logger.error(f"Unexpected error loading models and data: {e}")
        raise DataAccessError(f"Unexpected error loading models and data: {e}")
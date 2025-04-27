"""
Module for loading models and data.
"""
import os
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache

from app.exceptions import ModelLoadError, DataAccessError


class DataLoader:
    """Class to handle loading of models and data."""

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize with model directory.

        Args:
            model_dir: Directory where models and data are stored
        """
        self.model_dir = model_dir or os.path.dirname(os.path.abspath(__file__))

    @lru_cache(maxsize=2)
    def load_model(self, model_name: str) -> Any:
        """
        Load a model from disk with caching.

        Args:
            model_name: Name of the model file

        Returns:
            The loaded model

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            model_path = os.path.join(self.model_dir, model_name)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_name}: {e}")

    def load_all_models(self) -> Dict[str, Any]:
        """
        Load all required models.

        Returns:
            Dict: Dictionary mapping property types to models

        Raises:
            ModelLoadError: If any model loading fails
        """
        try:
            landed_model = self.load_model('landed_model.pkl')
            highrise_model = self.load_model('highrise_model.pkl')

            return {
                "0": landed_model,
                "1": highrise_model
            }
        except Exception as e:
            raise ModelLoadError(f"Failed to load models: {e}")

    @lru_cache(maxsize=1)
    def load_poi_data(self) -> Tuple[List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float]]]:
        """
        Load points of interest data.

        Returns:
            Tuple containing lists of coordinates for landmarks, stations, and malls

        Raises:
            DataAccessError: If data loading fails
        """
        try:
            poi_json_path = os.path.join(self.model_dir, "landmarks.json")

            with open(poi_json_path, "r") as f:
                poi_data = json.load(f)

            landmark_coords = [
                (item["Landmark_Latitude"], item["Landmark_Longitude"])
                for item in poi_data.get("Landmarks", [])
            ]
            station_coords = [
                (item["Stations_Latitude"], item["Stations_Longitude"])
                for item in poi_data.get("Stations", [])
            ]
            mall_coords = [
                (item["Mall_Latitude"], item["Mall_Longitude"])
                for item in poi_data.get("Malls", [])
            ]

            return landmark_coords, station_coords, mall_coords
        except Exception as e:
            raise DataAccessError(f"Failed to load POI data: {e}")
"""
Geospatial utilities for property valuation.
"""
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.exceptions import InvalidLocationError
from app.config import config


class GeoSpatialUtils:
    """Utilities for geospatial operations."""

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).

        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point

        Returns:
            float: Distance in kilometers
        """

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        radius = 6371

        return radius * c

    @staticmethod
    def get_coordinates(input_data: Dict) -> Tuple[float, float]:
        """
        Extract standardized latitude and longitude from input data.

        Args:
            input_data: Property data dictionary

        Returns:
            tuple: (latitude, longitude)

        Raises:
            InvalidLocationError: If coordinates cannot be found
        """
        lat = input_data.get('Latitude')
        lon = input_data.get('Longitude')

        if lat is None or lon is None:
            place = input_data.get("place", {})
            location = place.get("location", {})
            lat = location.get("lat")
            lon = location.get("lng")

        if lat is None or lon is None:
            raise InvalidLocationError(
                "Input must include 'Latitude' and 'Longitude' or nested place.location coordinates.")

        return float(lat), float(lon)

    @staticmethod
    def format_coordinates(input_data: Dict) -> Dict[str, float]:
        """
        Format latitude and longitude by truncating to specified decimal places.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Formatted coordinates
        """
        try:
            lat, lon = GeoSpatialUtils.get_coordinates(input_data)
            precision = 10 ** config.numeric_precision

            formatted_lat = int(float(lat) * precision) / precision
            formatted_lon = int(float(lon) * precision) / precision

            return {
                "formatted_Latitude": formatted_lat,
                "formatted_Longitude": formatted_lon
            }
        except (ValueError, TypeError, InvalidLocationError):
            return {"formatted_Latitude": 0, "formatted_Longitude": 0}

    @staticmethod
    def compute_poi_metrics(prop_lat: float, prop_lon: float,
                            landmark_coords: List[Tuple[float, float]],
                            station_coords: List[Tuple[float, float]],
                            mall_coords: List[Tuple[float, float]]) -> pd.Series:
        """
        Compute metrics related to points of interest (POIs) near a property.

        Args:
            prop_lat: Property latitude
            prop_lon: Property longitude
            landmark_coords: List of (lat, lon) tuples for landmarks
            station_coords: List of (lat, lon) tuples for transit stations
            mall_coords: List of (lat, lon) tuples for shopping malls

        Returns:
            Series: POI metrics including distances and counts
        """
        # Landmarks
        landmark_dists = [GeoSpatialUtils.haversine(prop_lat, prop_lon, lat, lon) for (lat, lon) in landmark_coords]
        if landmark_dists:
            nearest_landmark = min(landmark_dists)
            n_landmarks_3km = sum(1 for d in landmark_dists if d <= 3)
            n_landmarks_5km = sum(1 for d in landmark_dists if d <= 5)
            n_landmarks_10km = sum(1 for d in landmark_dists if d <= 10)
        else:
            nearest_landmark = np.nan
            n_landmarks_3km = n_landmarks_5km = n_landmarks_10km = 0

        # MRT Stations
        station_dists = [GeoSpatialUtils.haversine(prop_lat, prop_lon, lat, lon) for (lat, lon) in station_coords]
        mrt_within_1_5km = 1 if any(d <= 1.5 for d in station_dists) else 0
        mrt_within_3km = 1 if any(d <= 3 for d in station_dists) else 0
        mrt_within_5km = 1 if any(d <= 5 for d in station_dists) else 0

        # Malls
        mall_dists = [GeoSpatialUtils.haversine(prop_lat, prop_lon, lat, lon) for (lat, lon) in mall_coords]
        if mall_dists:
            nearest_mall = min(mall_dists)
            n_malls_3km = sum(1 for d in mall_dists if d <= 3)
            n_malls_5km = sum(1 for d in mall_dists if d <= 5)
        else:
            nearest_mall = np.nan
            n_malls_3km = n_malls_5km = 0

        return pd.Series({
            'distance_to_nearest_landmark': nearest_landmark,
            'n_landmarks_within_3km': n_landmarks_3km,
            'n_landmarks_within_5km': n_landmarks_5km,
            'n_landmarks_within_10km': n_landmarks_10km,
            '1.5_km_within_mrt': mrt_within_1_5km,
            '3_km_within_mrt': mrt_within_3km,
            '5_km_within_mrt': mrt_within_5km,
            'distance_to_nearest_mall': nearest_mall,
            'n_malls_within_3km': n_malls_3km,
            'n_malls_within_5km': n_malls_5km
        })
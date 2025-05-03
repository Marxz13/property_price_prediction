"""
Feature engineering utilities for property valuation.
"""
import re
from typing import Dict, List, Tuple

from app.models.property_types import PropertyTypeMappings
from app.models.geo_utils import GeoSpatialUtils


class FeatureEngineer:
    """Handles feature engineering for property data."""

    def __init__(self, landmark_coords: List[Tuple[float, float]],
                 station_coords: List[Tuple[float, float]],
                 mall_coords: List[Tuple[float, float]]):
        """
        Initialize with POI coordinates.

        Args:
            landmark_coords: List of landmark coordinates (lat, lon)
            station_coords: List of station coordinates (lat, lon)
            mall_coords: List of mall coordinates (lat, lon)
        """
        self.landmark_coords = landmark_coords
        self.station_coords = station_coords
        self.mall_coords = mall_coords

    def update_input_with_poi_features(self, input_data: Dict) -> Dict:
        """
        Update input data with point of interest (POI) features.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Updated property data with POI features
        """
        # Rest of your implementation
        lat, lon = GeoSpatialUtils.get_coordinates(input_data)

        # Update input with standardized coordinates
        input_data["Latitude"] = lat
        input_data["Longitude"] = lon

        # Compute POI metrics
        metrics = GeoSpatialUtils.compute_poi_metrics(
            lat, lon, self.landmark_coords, self.station_coords, self.mall_coords)

        # Select key POI features to include
        poi_features = {
            'n_landmarks_within_3km': metrics['n_landmarks_within_3km'],
            'n_landmarks_within_5km': metrics['n_landmarks_within_5km'],
            'n_landmarks_within_10km': metrics['n_landmarks_within_10km'],
            'distance_to_nearest_landmark': metrics['distance_to_nearest_landmark'],
            '1.5_km_within_mrt': metrics['1.5_km_within_mrt'],
            '3_km_within_mrt': metrics['3_km_within_mrt'],
            '5_km_within_mrt': metrics['5_km_within_mrt'],
            'distance_to_nearest_mall': metrics['distance_to_nearest_mall'],
            'n_malls_within_3km': metrics['n_malls_within_3km'],
            'n_malls_within_5km': metrics['n_malls_within_5km']
        }

        input_data.update(poi_features)
        return input_data

    @staticmethod
    def extract_floors_from_input(input_data: Dict) -> int:
        """
        Extract floors information from input data.

        Args:
            input_data: Property data dictionary

        Returns:
            int: Number of floors
        """
        # Try to find floors in propertyDetails$2 first
        details2 = input_data.get("propertyDetails$2", {})
        floors = details2.get("floorCount") or details2.get("floors")

        # If not found, try propertyDetails$1
        if floors is None:
            details1 = input_data.get("propertyDetails$1", {})
            floors = details1.get("floorCount") or details1.get("floors")

        # If still not found, default to 1
        return int(floors) if floors is not None else 1

    @staticmethod
    def extract_area_encoded(input_data: Dict) -> int:
        """
        Extract encoded area value from address.

        Args:
            input_data: Property data dictionary

        Returns:
            int: Encoded area value (0 if not found)
        """
        formatted_address = input_data.get("place", {}).get("formattedAddress", "").lower()

        for area, code in PropertyTypeMappings.AREA_MAP.items():
            if area in formatted_address:
                return code

        return 0

    @staticmethod
    def derive_structure_features(input_data: Dict) -> Dict[str, int]:
        """
        Extract structure-related features from property data.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Structure features including solar, private_pool, and lift
        """
        # Extract the structure list (ensure it is a list)
        struct_list = input_data.get("others", {}).get("structure", {}).get("structure", [])
        if not isinstance(struct_list, list):
            struct_list = []

        # "0" => septic system (ignore),
        # "1" => solar panels,
        # "2" => private pool,
        # "4" => private lift
        solar = 1 if "1" in struct_list else 0
        private_pool = 1 if "2" in struct_list else 0
        lift = 1 if "4" in struct_list else 0

        return {"solar": solar, "private_pool": private_pool, "lift": lift}

    @staticmethod
    def get_size_features(input_data: Dict) -> Dict[str, float]:
        """
        Extract size-related features from property data.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Size features including built-up and land areas
        """
        details = input_data.get("propertyDetails$1", {})

        return {
            "built_up_in_sqft": float(details.get("buildUpAreaInSqft", 0) or 0),
            "land_area_in_sqft": float(details.get("landAreaInSqft", 0) or 0)
        }

    @staticmethod
    def get_extended_feature(input_data: Dict) -> int:
        """
        Determine if property has extension based on structure extension value.

        Args:
            input_data: Property data dictionary

        Returns:
            int: 1 if extended, 0 if not
        """
        extension = input_data.get("others", {}).get("structure", {}).get("structureExtension")

        if extension is None:
            return 0

        return 1 if re.search(r'\d', str(extension)) else 0

    @staticmethod
    def get_bed_bath_features(input_data: Dict) -> Dict[str, int]:
        """
        Extract bedroom and bathroom counts from property data.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Bedroom and bathroom counts
        """
        details = input_data.get("propertyDetails$2", {})

        return {
            "bedroomCount": int(details.get("bedroomCount", 0) or 0),
            "bathroomCount": int(details.get("bathroomCount", 0) or 0)
        }

    @staticmethod
    def get_furnishing_encoded(input_data: Dict) -> int:
        """
        Determine furnishing level based on livingRoom value.

        Args:
            input_data: Property data dictionary

        Returns:
            int: Furnishing encoded value:
                - 2: Fully furnished (livingRoom="3" or "4")
                - 1: Partially furnished (livingRoom="2")
                - 3: Unfurnished (livingRoom="1")
                - 0: Default or unknown
        """
        living_room = input_data.get("livingRoom", "0")

        if living_room in ["3", "4"]:
            return 2  # Fully furnished
        elif living_room == "2":
            return 1  # Partially furnished
        elif living_room == "1":
            return 3  # Unfurnished
        else:
            return 0  # Default or unknown

    @staticmethod
    def process_foundation_issues(input_data: Dict) -> int:
        """
        Check if property has foundation issues.

        Args:
            input_data: Property data dictionary

        Returns:
            int: 1 if foundation issues exist, 0 if not
        """
        try:
            # Check if foundation issue exists in the others.issues.otherIssues array
            other_issues = input_data.get("others", {}).get("issues", {}).get("otherIssues", [])
            return 1 if "0" in other_issues else 0
        except (TypeError, KeyError):
            return 0

    @staticmethod
    def process_extension_features(input_data: Dict) -> Dict[str, int]:
        """
        Process extension-related features.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Extension features
        """
        partial_extended = 0
        full_extended = 0

        try:
            # Check structureExtension value
            extension = input_data.get("others", {}).get("structure", {}).get("structureExtension")
            if extension in ["0", "2"]:  # Ground floor extensions
                partial_extended = 1
            elif extension in ["1", "3", "4"]:  # Full extensions or additional storey
                full_extended = 1
        except (TypeError, KeyError):
            pass

        return {
            "partial_extended": partial_extended,
            "full_extended": full_extended
        }

    @staticmethod
    def process_floor_types(input_data: Dict) -> Dict[str, int]:
        """
        Process main floor types from flooringTypes.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Floor type features
        """
        main_floor_type = 0
        main_floor_type_2 = 0

        try:
            flooring_types = input_data.get("flooringTypes", [])
            if flooring_types and len(flooring_types) > 0:
                main_floor_type = int(flooring_types[0])
                if len(flooring_types) > 1:
                    main_floor_type_2 = int(flooring_types[1])
        except (ValueError, IndexError, TypeError):
            pass

        return {
            "main_floor_type": main_floor_type,
            "main_floor_type_2": main_floor_type_2
        }

    @staticmethod
    def swap_security_value(input_data: Dict) -> Dict:
        """
        Swap security value for proper encoding.

        Args:
            input_data: Property data dictionary

        Returns:
            Dict: Updated property data
        """
        val = input_data.get("security")

        if val == "0":
            input_data["security"] = "1"
        elif val == "1":
            input_data["security"] = "0"

        return input_data
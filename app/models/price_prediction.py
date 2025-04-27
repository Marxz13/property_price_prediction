"""
Property price prediction module.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from app.models.property_types import PropertyTypeClassifier
from app.models.geo_utils import GeoSpatialUtils
from app.models.feature_engineering import FeatureEngineer
from app.exceptions import PredictionError, InvalidPropertyTypeError


class PricePredictor:
    """Class for predicting property prices."""

    def __init__(self, models: Dict[str, Any],
                 landmark_coords: List[Tuple[float, float]],
                 station_coords: List[Tuple[float, float]],
                 mall_coords: List[Tuple[float, float]]):
        """
        Initialize the price predictor.

        Args:
            models: Dictionary mapping property types to prediction models
            landmark_coords: Coordinates of landmarks
            station_coords: Coordinates of transit stations
            mall_coords: Coordinates of shopping malls
        """
        self.models = models
        self.landmark_coords = landmark_coords
        self.station_coords = station_coords
        self.mall_coords = mall_coords
        self.feature_engineer = FeatureEngineer(landmark_coords, station_coords, mall_coords)

    def prepare_features_for_prediction(self, input_data: Dict) -> pd.DataFrame:
        """
        Prepare features for property price prediction.

        Args:
            input_data: Property data dictionary

        Returns:
            DataFrame: Features ready for model prediction
        """
        processed = input_data.copy()

        processed = self.feature_engineer.update_input_with_poi_features(processed)

        processed = self.feature_engineer.swap_security_value(processed)

        processed["area_encoded"] = self.feature_engineer.extract_area_encoded(processed)

        processed.update(self.feature_engineer.derive_structure_features(processed))

        processed["propertyType_encoded"] = PropertyTypeClassifier.map_property_type(processed)

        processed.update(self.feature_engineer.get_bed_bath_features(processed))

        processed["extended"] = self.feature_engineer.get_extended_feature(processed)

        processed.update(self.feature_engineer.get_size_features(processed))

        processed["propertyCategory_encoded"] = PropertyTypeClassifier.get_property_category(processed)

        processed["furnishing_encoded"] = self.feature_engineer.get_furnishing_encoded(processed)

        processed["is_landed"] = PropertyTypeClassifier.is_property_landed(processed)

        processed.update(GeoSpatialUtils.format_coordinates(processed))

        processed["foundation_issues"] = self.feature_engineer.process_foundation_issues(processed)

        processed.update(self.feature_engineer.process_extension_features(processed))

        processed.update(self.feature_engineer.process_floor_types(processed))

        feature_cols = [
            "propertyType_encoded",
            "built_up_in_sqft",
            "land_area_in_sqft",
            "formatted_Latitude",
            "formatted_Longitude",
            "bathroomCount",
            "bedroomCount",
            "area_encoded",
            "private_pool",
            "gated_guarded",
            "solar",
            "lift",
            "n_landmarks_within_3km",
            "1.5_km_within_mrt",
            "3_km_within_mrt",
            "5_km_within_mrt",
            "n_malls_within_3km",
            "partial_extended",
            "full_extended",
            "extended",
            "main_floor_type_2",
            "kitchen",
            "bathroom",
            "livingRoom",
            "main_floor_type",
            "foundation_issues",
            "homeExterior",
            "view",
            "is_landed",
            "furnishing_encoded",
            "propertyCategory_encoded",
        ]

        defaults = {
            "gated_guarded": processed.get("security", 0),
            "extended": 0,
            "partial_extended": 0,
            "full_extended": 0,
            "main_floor_type": 0,
            "main_floor_type_2": 0,
            "foundation_issues": 0,
            "homeExterior": 0,
            "view": 0,
            "kitchen": 2,
            "bathroom": 2,
            "livingRoom": 2,
        }


        for key, default_val in defaults.items():
            if key not in processed:
                processed[key] = default_val

        features_dict = {col: processed.get(col, defaults.get(col, 0)) for col in feature_cols}

        df_features = pd.DataFrame([features_dict], columns=feature_cols)

        for col in feature_cols:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)

        return df_features

    def predict_price(self, input_data: Dict) -> float:
        """
        Predict property price based on input data.

        Args:
            input_data: Property data dictionary

        Returns:
            float: Predicted property price

        Raises:
            PredictionError: If prediction process fails
            InvalidPropertyTypeError: If property type is invalid
        """
        try:
            property_type_key = input_data.get("propertyType$1")
            if property_type_key not in self.models:
                raise InvalidPropertyTypeError(
                    f"Invalid property type: {property_type_key}. Expected '0' (landed) or '1' (highrise).")

            model = self.models[property_type_key]

            df_features = self.prepare_features_for_prediction(input_data)

            y_pred_log = model.predict(df_features)

            predicted_price = np.exp(y_pred_log)[0]

            return predicted_price

        except InvalidPropertyTypeError as e:
            raise
        except Exception as e:
            raise PredictionError(f"Error predicting property price: {e}")

    @staticmethod
    def round_cents_to_thousand(price_in_cents: int) -> int:
        """
        Convert cents to dollars, round to the nearest thousand dollars,
        then convert back to cents.

        Args:
            price_in_cents: Price in cents

        Returns:
            int: Rounded price in cents

        Example:
            5234549 cents (~52345.49 dollars) becomes 5200000 cents.
        """
        dollars = price_in_cents / 100.0
        rounded_dollars = round(dollars / 1000) * 1000
        return int(rounded_dollars * 100)
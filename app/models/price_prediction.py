"""
Property price prediction module.
"""
import json

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

        print("=== INPUT DATA ===")
        print(json.dumps(processed, indent=2, default=str))

        # Process POI features
        processed = self.feature_engineer.update_input_with_poi_features(processed)

        # Extract core features from input
        processed["propertyType_encoded"] = PropertyTypeClassifier.map_property_type(processed)
        processed.update(self.feature_engineer.get_bed_bath_features(processed))
        processed.update(self.feature_engineer.get_size_features(processed))
        processed["propertyCategory_encoded"] = PropertyTypeClassifier.get_property_category(processed)

        if 'furnishing_encoded' not in input_data or input_data['furnishing_encoded'] is None:
            input_data['furnishing_encoded'] = FeatureEngineer.get_furnishing_encoded(input_data)

        processed["tenure_encoded"] = processed.get("tenure_encoded")
        processed["propertyTitle_encoded"] = processed.get("propertyTitle_encoded")
        processed["propertyClass_encoded"] = processed.get("propertyClass_encoded")

        # Fix coordinate naming
        formatted_coords = GeoSpatialUtils.format_coordinates(processed)
        processed["latitude"] = formatted_coords["formatted_Latitude"]
        processed["longitude"] = formatted_coords["formatted_Longitude"]

        # Distance to nearest mall (POI feature)
        processed["distance_to_nearest_mall"] = processed.get("distance_to_nearest_mall")

        # n_landmarks_within_10km (may need to calculate if not in POI metrics)
        processed["n_landmarks_within_10km"] = processed.get("n_landmarks_within_10km")

        # Calculate derived features
        processed["log_built_up"] = np.log1p(float(processed.get("built_up_in_sqft")))

        bed_count = float(processed.get("bedroomCount"))
        bath_count = float(processed.get("bathroomCount"))

        processed["bed_bath_ratio"] = bed_count / max(bath_count, 1)  # Avoid division by zero
        processed["bed_bath_product"] = bed_count * bath_count
        processed["sqft_per_bedroom"] = float(processed.get("built_up_in_sqft", 0)) / max(bed_count, 1)

        # Coordinate transformations
        lat = processed["latitude"]
        lon = processed["longitude"]
        processed["lat_sq"] = lat * lat
        processed["long_sq"] = lon * lon
        processed["lat_long"] = lat * lon

        # Add landed property features if applicable
        if PropertyTypeClassifier.is_property_landed(processed):
            processed.update(self.feature_engineer.derive_structure_features(processed))
            processed["extended"] = self.feature_engineer.get_extended_feature(processed)
            processed["gated_guarded"] = processed.get("security", 0)
        else:
            # Default values for landed features when not applicable
            for feature in ["private_pool", "gated_guarded", "solar", "lift", "extended"]:
                processed[feature] = 0

        # Define the exact feature columns needed by the model
        model_feature_cols = [
            'propertyType_encoded',
            'bedroomCount',
            'bathroomCount',
            'furnishing_encoded',
            'tenure_encoded',
            'propertyTitle_encoded',
            'propertyCategory_encoded',
            'built_up_in_sqft',
            'propertyClass_encoded',
            'latitude',
            'longitude',
            'n_landmarks_within_10km',
            '1.5_km_within_mrt',
            '3_km_within_mrt',
            'distance_to_nearest_mall',
            'n_malls_within_3km',
            'log_built_up',
            'bed_bath_ratio',
            'bed_bath_product',
            'lat_sq',
            'long_sq',
            'lat_long',
            'sqft_per_bedroom'
        ]

        # For landed properties, add these features
        if PropertyTypeClassifier.is_property_landed(processed):
            model_feature_cols.extend([
                'private_pool',
                'gated_guarded',
                'solar',
                'lift',
                'extended',
                'land_area_in_sqft'
            ])

        # Create DataFrame with only the required features
        features_dict = {col: processed.get(col, 0) for col in model_feature_cols}
        df_features = pd.DataFrame([features_dict], columns=model_feature_cols)

        print("=== GENERATED FEATURES ===")
        feature_cols = list(df_features.columns)
        print(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}")

        # Print first few rows of features with values
        print("\nFeature values:")
        print(df_features.head(1).to_string())

        # Check for NaN values
        nan_counts = df_features.isna().sum()
        if nan_counts.sum() > 0:
            print("\nWARNING: NaN values detected in features:")
            print(nan_counts[nan_counts > 0].to_string())

        for col in model_feature_cols:
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

    def calculate_confidence_score(self, input_data: Dict, predicted_price: float,
                                   reports_data: pd.DataFrame = None) -> float:
        """
        Calculate a dynamic confidence score based on multiple factors.

        Args:
            input_data: Property data dictionary
            predicted_price: The predicted price value
            reports_data: DataFrame of property reports (if available)

        Returns:
            float: Confidence score between 0 and 1
        """
        # Start with a baseline confidence
        base_confidence = 0.75

        # Initialize adjustment factors
        data_completeness_factor = 0
        spatial_confidence_factor = 0
        temporal_confidence_factor = 0
        property_similarity_factor = 0
        model_certainty_factor = 0

        # ---- 1. Data completeness factor ----
        required_fields = [
            "propertyType$1", "propertyType$2", "propertyDetails$1",
            "built_up_in_sqft", "bedroomCount", "bathroomCount"
        ]

        optional_fields = [
            "propertyTitle_encoded", "tenure_encoded", "gated_guarded",
            "homeExterior", "propertyClass_encoded"
        ]

        # Check required fields (max penalty: -0.2)
        missing_required = sum(1 for field in required_fields if not self._check_field_exists(input_data, field))
        required_penalty = min(missing_required * 0.05, 0.2)

        # Check optional fields (max bonus: +0.05)
        present_optional = sum(1 for field in optional_fields if self._check_field_exists(input_data, field))
        optional_bonus = min(present_optional * 0.01, 0.05)

        data_completeness_factor = -required_penalty + optional_bonus

        # ---- 2. Spatial confidence factor ----
        # Higher confidence for properties with more comparables nearby
        try:
            lat, lon = GeoSpatialUtils.get_coordinates(input_data)

            # If we have reports data, calculate spatial confidence
            if reports_data is not None and not reports_data.empty:
                # Only proceed if reports data has lat/long
                if 'latitude' in reports_data.columns and 'longitude' in reports_data.columns:
                    # Calculate distances to all properties in reports
                    distances = []
                    for idx, row in reports_data.iterrows():
                        if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                            dist = GeoSpatialUtils.haversine(
                                lat, lon, float(row['latitude']), float(row['longitude'])
                            )
                            distances.append(dist)

                    # Count properties within various distances
                    props_within_1km = sum(1 for d in distances if d <= 1)
                    props_within_3km = sum(1 for d in distances if d <= 3)
                    props_within_5km = sum(1 for d in distances if d <= 5)

                    # Calculate spatial confidence factor (-0.1 to +0.1)
                    if props_within_1km >= 10:
                        spatial_confidence_factor = 0.1  # Excellent data density
                    elif props_within_1km >= 5:
                        spatial_confidence_factor = 0.07
                    elif props_within_3km >= 10:
                        spatial_confidence_factor = 0.05
                    elif props_within_3km >= 5:
                        spatial_confidence_factor = 0.03
                    elif props_within_5km >= 10:
                        spatial_confidence_factor = 0.01
                    elif props_within_5km < 3:
                        spatial_confidence_factor = -0.05  # Sparse data
                    else:
                        spatial_confidence_factor = 0
        except Exception:
            # If we can't calculate spatial confidence, default to 0
            spatial_confidence_factor = 0

        # ---- 3. Property similarity factor ----
        # Higher confidence for common property configurations
        property_type = input_data.get("propertyType$1")
        bedroom_count = float(input_data.get("propertyDetails$2", {}).get("bedroomCount", 0))
        bathroom_count = float(input_data.get("propertyDetails$2", {}).get("bathroomCount", 0))
        built_up_area = float(input_data.get("propertyDetails$1", {}).get("buildUpAreaInSqft", 0))

        # Determine if property attributes are common or unusual
        unusual_attributes = 0

        # Check bedroom count (most common: 2-4)
        if bedroom_count < 1 or bedroom_count > 6:
            unusual_attributes += 1

        # Check bathroom count (most common: 1-3)
        if bathroom_count < 1 or bathroom_count > 5:
            unusual_attributes += 1

        # Check built-up area (depends on property type)
        if property_type == "0":  # Landed
            if built_up_area < 800 or built_up_area > 5000:
                unusual_attributes += 1
        else:  # Highrise
            if built_up_area < 500 or built_up_area > 3000:
                unusual_attributes += 1

        # Calculate property similarity factor (-0.1 to +0.05)
        if unusual_attributes == 0:
            property_similarity_factor = 0.05  # Common property configuration
        elif unusual_attributes == 1:
            property_similarity_factor = 0
        elif unusual_attributes == 2:
            property_similarity_factor = -0.05
        else:
            property_similarity_factor = -0.1  # Very unusual property

        # ---- 4. Model certainty factor ----
        # Check if the predicted price is in a reasonable range
        if reports_data is not None and not reports_data.empty and 'asking_price' in reports_data.columns:
            # Convert predicted_price from cents to dollars for comparison
            predicted_price_dollars = predicted_price / 100

            # Calculate median price in reports data
            median_price = reports_data['asking_price'].median()

            # Calculate the deviation from median
            if median_price > 0:
                deviation_pct = abs(predicted_price_dollars - median_price) / median_price

                if deviation_pct < 0.1:
                    model_certainty_factor = 0.05  # Low deviation
                elif deviation_pct < 0.2:
                    model_certainty_factor = 0.02
                elif deviation_pct < 0.3:
                    model_certainty_factor = 0
                elif deviation_pct < 0.5:
                    model_certainty_factor = -0.05
                else:
                    model_certainty_factor = -0.1  # High deviation

        # ---- 5. Temporal confidence factor ----
        # Higher confidence for areas with recent data
        if reports_data is not None and not reports_data.empty and 'date_of_asking_price' in reports_data.columns:
            try:
                # Convert date strings to datetime objects
                reports_data['date_parsed'] = pd.to_datetime(
                    reports_data['date_of_asking_price'],
                    errors='coerce'
                )

                # Filter valid dates and check recency
                valid_dates = reports_data['date_parsed'].dropna()

                if not valid_dates.empty:
                    # Calculate days since most recent data
                    now = pd.Timestamp.now()
                    days_since_most_recent = (now - valid_dates.max()).days

                    # Calculate temporal confidence factor (-0.1 to +0.05)
                    if days_since_most_recent < 30:
                        temporal_confidence_factor = 0.05  # Very recent data
                    elif days_since_most_recent < 90:
                        temporal_confidence_factor = 0.03
                    elif days_since_most_recent < 180:
                        temporal_confidence_factor = 0
                    elif days_since_most_recent < 365:
                        temporal_confidence_factor = -0.05
                    else:
                        temporal_confidence_factor = -0.1  # Very old data
            except Exception:
                # If we can't calculate temporal confidence, default to 0
                temporal_confidence_factor = 0

        # ---- Combine all factors ----
        # Apply all adjustment factors to base confidence
        confidence = base_confidence + \
                     data_completeness_factor + \
                     spatial_confidence_factor + \
                     property_similarity_factor + \
                     model_certainty_factor + \
                     temporal_confidence_factor

        # Ensure confidence stays within reasonable bounds (0.5 to 0.95)
        confidence = max(min(confidence, 0.95), 0.5)

        # Round to 2 decimal places
        return round(confidence, 2)

    def _check_field_exists(self, data: Dict, field: str) -> bool:
        """
        Check if a field exists and has a non-empty value in the data.

        Args:
            data: Dictionary to check
            field: Field name to check (supports nested fields with $ separator)

        Returns:
            bool: True if field exists and has value, False otherwise
        """
        if '$' in field:
            # Handle nested fields
            parts = field.split('$')
            current = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            return current is not None and current != ""
        else:
            # Handle regular fields
            if field in data:
                return data[field] is not None and data[field] != ""
            return False
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
"""
Service layer for property price prediction.
"""
from typing import Dict, List, Tuple, Any
import pandas as pd
from flask import current_app

from app.models.price_prediction import PricePredictor
from app.models.geo_utils import GeoSpatialUtils
from app.exceptions import PredictionError, InvalidPropertyTypeError


def predict_property_price(input_data: Dict, models: Dict[str, Any],
                           poi_data: Tuple[List[Tuple[float, float]],
                           List[Tuple[float, float]],
                           List[Tuple[float, float]]]) -> Dict:
    """
    Predict property price based on input data.

    Args:
        input_data: Property data dictionary
        models: Dictionary mapping property types to models
        poi_data: Tuple of landmark, station, and mall coordinates

    Returns:
        Dict: Prediction results including price and confidence metrics
    """
    try:
        landmark_coords, station_coords, mall_coords = poi_data

        price_predictor = PricePredictor(models, landmark_coords, station_coords, mall_coords)

        predicted_price = price_predictor.predict_price(input_data)

        rounded_price = PricePredictor.round_cents_to_thousand(int(predicted_price * 100))

        result = {
            "predicted_price": rounded_price,
            "predicted_price_raw": int(predicted_price * 100),
            "confidence_score": 0.85,
            "prediction_date": pd.Timestamp.now().isoformat()
        }

        property_type_encoded = input_data.get("propertyType_encoded", None)
        if property_type_encoded is not None:
            result["property_type_encoded"] = property_type_encoded

        try:
            lat, lon = GeoSpatialUtils.get_coordinates(input_data)
            result["coordinates"] = {
                "latitude": lat,
                "longitude": lon
            }
        except Exception:
            pass

        return result

    except InvalidPropertyTypeError as e:
        # Re-raise with more context
        raise InvalidPropertyTypeError(f"Invalid property type in input data: {e}")
    except Exception as e:
        # Log error and re-raise as PredictionError
        current_app.logger.error(f"Error predicting property price: {e}")
        raise PredictionError(f"Error predicting property price: {e}")
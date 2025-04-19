"""
Service layer for market analysis functionality.
"""
from typing import Dict, List
from flask import current_app

from app.models.market_analysis import MarketAnalyzer
from app.models.response_builder import ResponseBuilder, PropertyComparator
from app.models.price_prediction import PricePredictor
from app.exceptions import PropertyValuationError, InvalidLocationError
from app.services.data_loader import get_db_client, load_models_and_data


def get_market_analysis(input_data: Dict) -> Dict:
    """
    Get comprehensive market analysis for a property.

    Args:
        input_data: Property data dictionary

    Returns:
        Dict: Market analysis results
    """
    try:
        # Get database client
        db_client = get_db_client()

        # Load models and data
        models, poi_data = load_models_and_data()

        # Unpack POI data
        landmark_coords, station_coords, mall_coords = poi_data

        market_analyzer = MarketAnalyzer()
        price_predictor = PricePredictor(models, landmark_coords, station_coords, mall_coords)

        transaction_data = market_analyzer.get_transaction_data()

        # Get address components
        address_components = input_data.get("place", {}).get("addressComponents", [])

        # Predict price
        predicted_price = price_predictor.predict_price(input_data)

        # Create response builder
        response_builder = ResponseBuilder(db_client, price_predictor, market_analyzer)

        # Build response
        response = response_builder.get_response_json(
            transaction_data,
            predicted_price,
            address_components,
            input_data
        )

        return response

    except InvalidLocationError as e:
        raise InvalidLocationError(f"Invalid location in input data: {e}")
    except Exception as e:
        current_app.logger.error(f"Error in market analysis: {e}")
        raise PropertyValuationError(f"Error performing market analysis: {e}")


def get_comparable_properties(input_data: Dict, limit: int = 5) -> List[Dict]:
    """
    Get comparable properties for a given property.

    Args:
        input_data: Property data dictionary
        limit: Maximum number of comparable properties to return

    Returns:
        List[Dict]: List of comparable properties
    """
    try:
        # Get database client
        db_client = get_db_client()

        # Get property comparator
        property_comparator = PropertyComparator(db_client)

        # Extract information needed for comparison
        area = None
        postal_code = None

        # Check for address components
        address_components = input_data.get("place", {}).get("addressComponents", [])

        # Extract area and postal code
        for component in address_components:
            component_types = component.get("types", [])
            if "sublocality_level_1" in component_types or "area" in component_types:
                area = component.get("longText")
            elif "postal_code" in component_types:
                postal_code = component.get("longText")

        # If area or postal code is not found, try to extract from other fields
        if not area:
            area = input_data.get("area")

        if not postal_code:
            postal_code = input_data.get("postal_code")

        # If we still don't have location information, raise an error
        if not area and not postal_code:
            raise InvalidLocationError("Location information not found in input data")

        # Get property type
        from app.models.property_types import PropertyTypeClassifier
        try:
            property_type = PropertyTypeClassifier.map_property_type(input_data)
        except Exception:
            property_type = None

        # Get comparable properties
        comparable_properties = property_comparator.get_last_5_sold_properties(
            input_data,
            area,
            postal_code,
            property_type
        )

        # Limit the number of properties if needed
        if limit > 0 and len(comparable_properties) > limit:
            comparable_properties = comparable_properties[:limit]

        return comparable_properties

    except InvalidLocationError as e:
        raise InvalidLocationError(f"Invalid location in input data: {e}")
    except Exception as e:
        current_app.logger.error(f"Error getting comparable properties: {e}")
        raise PropertyValuationError(f"Error getting comparable properties: {e}")
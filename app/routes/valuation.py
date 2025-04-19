"""
Route handlers for property valuation API endpoints.
"""
from flask import Blueprint, request, jsonify, current_app
import json

from app.services.price_prediction import predict_property_price
from app.services.data_loader import load_models_and_data
from app.exceptions import PropertyValuationError

bp = Blueprint("valuation", __name__, url_prefix="/api/valuation")


@bp.route("/predict", methods=["POST"])
def predict_price():
    """
    Predict property value based on input data.

    Expected JSON body: Property details including location, features, etc.

    Returns:
        JSON response with predicted price and confidence score
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Load models and data
        models, poi_data = load_models_and_data()

        # Predict price
        result = predict_property_price(data, models, poi_data)

        return jsonify(result), 200

    except PropertyValuationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error in price prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/bulk-predict", methods=["POST"])
def bulk_predict():
    """
    Predict property values for multiple properties.

    Expected JSON body: List of property details

    Returns:
        JSON response with list of predictions
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data or not isinstance(data, list):
            return jsonify({"error": "Input must be a list of properties"}), 400

        # Load models and data
        models, poi_data = load_models_and_data()

        # Process each property
        results = []
        for property_data in data:
            try:
                result = predict_property_price(property_data, models, poi_data)
                results.append(result)
            except PropertyValuationError as e:
                results.append({"error": str(e), "input": property_data})

        return jsonify(results), 200

    except Exception as e:
        current_app.logger.error(f"Error in bulk prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500
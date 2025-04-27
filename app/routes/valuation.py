"""
Route handlers for property valuation API endpoints.
"""
from flask import Blueprint, request, jsonify, current_app
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
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        models, poi_data = load_models_and_data()
        result = predict_property_price(data, models, poi_data)

        return jsonify(result), 200

    except PropertyValuationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error in price prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

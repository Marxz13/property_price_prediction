"""
Route handlers for property valuation API endpoints.
"""
from flask import Blueprint, request, jsonify, current_app
from app.services.price_prediction import predict_property_price
from app.services.data_loader import load_models_and_data
from app.exceptions import PropertyValuationError
import json

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

        # Debug: Print and log the payload
        print("=== RECEIVED PAYLOAD ===")
        print(json.dumps(data, indent=2))
        print("========================")

        # Also log to Flask logger
        current_app.logger.info(f"Received payload: {json.dumps(data)}")

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        models, poi_data = load_models_and_data()
        result = predict_property_price(data, models, poi_data)

        # Debug: Print the prediction result
        print("=== PREDICTION RESULT ===")
        print(json.dumps(result, indent=2))
        print("=========================")

        return jsonify(result), 200

    except PropertyValuationError as e:
        error_msg = str(e)
        print(f"=== VALIDATION ERROR ===\n{error_msg}\n======================")
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        error_msg = str(e)
        print(f"=== INTERNAL ERROR ===\n{error_msg}\n====================")
        current_app.logger.error(f"Error in price prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500
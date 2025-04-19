"""
Route handlers for market analysis API endpoints.
"""
from flask import Blueprint, request, jsonify, current_app

from app.services.market_analysis import get_market_analysis, get_comparable_properties
from app.exceptions import PropertyValuationError, InvalidLocationError

bp = Blueprint("market_analysis", __name__, url_prefix="/api/market")


@bp.route("/analysis", methods=["POST"])
def analyze_market():
    """
    Get comprehensive market analysis for a property.

    Expected JSON body: Property details including location

    Returns:
        JSON response with market trends, comparable properties, etc.
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Get market analysis
        result = get_market_analysis(data)

        return jsonify(result), 200

    except InvalidLocationError as e:
        return jsonify({"error": str(e)}), 400
    except PropertyValuationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error in market analysis: {e}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/comparables", methods=["POST"])
def get_comparables():
    """
    Get comparable properties for a given property.

    Expected JSON body: Property details including location

    Returns:
        JSON response with list of comparable properties
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Get number of comparables to return
        limit = request.args.get("limit", 5, type=int)

        # Get comparable properties
        result = get_comparable_properties(data, limit)

        return jsonify(result), 200

    except InvalidLocationError as e:
        return jsonify({"error": str(e)}), 400
    except PropertyValuationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error getting comparable properties: {e}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/trends", methods=["GET"])
def get_market_trends():
    """
    Get market trends for a specific area.

    Query parameters:
        postal_code: Postal code of the area
        period: Time period for trends (e.g., "1m", "3m", "6m", "1y")

    Returns:
        JSON response with market trends
    """
    try:
        # Get query parameters
        postal_code = request.args.get("postal_code", type=str)
        period = request.args.get("period", "6m", type=str)

        if not postal_code:
            return jsonify({"error": "Postal code is required"}), 400

        # Get market trends
        # Note: This function would need to be implemented
        result = {"message": "This endpoint is not fully implemented yet"}

        return jsonify(result), 200

    except Exception as e:
        current_app.logger.error(f"Error getting market trends: {e}")
        return jsonify({"error": "Internal server error"}), 500
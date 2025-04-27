"""
Response generation for property valuation API.
"""
from typing import Dict

from app.models.price_prediction import PricePredictor

class ResponseBuilder:
    """Build response for property valuation API."""

    def __init__(self):
        """
        Initialize
        """

    @staticmethod
    def get_response_json(predicted_result: float,) -> Dict:
        """
        Generate comprehensive market analysis response.

        Args:
            predicted_result: Predicted price from model
        Returns:
            Dict: Complete market analysis response
        """
        return {
            "value_estimate": PricePredictor.round_cents_to_thousand(int(float(predicted_result) * 100)),
        }

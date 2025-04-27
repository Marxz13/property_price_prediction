"""
Market analysis utilities for property valuation.
"""
import pandas as pd
import numpy as np
from typing import Dict

from app.models.property_types import PropertyTypeClassifier
from app.config import config


class MarketAnalyzer:
    """Class for analyzing property market data."""

    def __init__(self):
        """Initialize market analyzer with database connections."""
        self.db_client = config.get_mongodb_client()

    def get_reports_data(self, limit=1000) -> pd.DataFrame:
        """
        Retrieve property data from MongoDB reports database.

        Args:
            limit: Maximum number of records to retrieve

        Returns:
            DataFrame: Property data
        """
        collection = self.db_client["reports"]
        cursor = collection.find().limit(limit)
        return pd.DataFrame(list(cursor))

    @staticmethod
    def calc_median_price(sold_prices: pd.Series) -> float:
        """
        Calculate median price from a series of prices.

        Args:
            sold_prices: Series of sold prices

        Returns:
            float: Median price or 0 if no data
        """
        if sold_prices.empty:
            return 0

        # If any element is a list or array, explode the series
        if sold_prices.apply(lambda x: isinstance(x, (list, np.ndarray))).any():
            sold_prices = sold_prices.explode()

        numeric_prices = pd.to_numeric(sold_prices, errors='coerce')
        return numeric_prices.median()

    @staticmethod
    def calculate_recommended_selling_price( reports_data: pd.DataFrame,
                                           predicted_price: float,
                                           input_data: Dict) -> int:
        """
        Calculate an enhanced recommended selling price using multiple factors.

        Args:
            reports_data: Property data
            predicted_price: Base predicted price from the ML model
            input_data: Property details and features

        Returns:
            int: Recommended price in cents
        """
        # Start with the predicted price as our base
        base_price = float(predicted_price)

        # Extract key property attributes for segmentation
        try:
            property_type_encoded = PropertyTypeClassifier.map_property_type(input_data)
        except Exception:
            property_type_encoded = 0

        property_category = PropertyTypeClassifier.get_property_category(input_data)
        built_up_area = input_data.get("propertyDetails$1", {}).get("buildUpAreaInSqft", 0)
        land_area = input_data.get("propertyDetails$1", {}).get("landAreaInSqft", 0)
        bedrooms = input_data.get("propertyDetails$2", {}).get("bedroomCount", 0)
        bathrooms = input_data.get("propertyDetails$2", {}).get("bathroomCount", 0)

        # Get location information
        from feature_engineering import FeatureEngineer
        area_encoded = FeatureEngineer.extract_area_encoded(input_data)
        location_data = input_data.get("place", {})
        lat = location_data.get("location", {}).get("lat")
        lng = location_data.get("location", {}).get("lng")

        # 1. SEGMENTATION - Find truly comparable properties
        segment_adjustment = 1.0

        if not reports_data.empty:
            # Convert numeric columns
            numeric_cols = ['propertyType_encoded', 'propertyCategory_encoded', 'built_up_in_sqft',
                            'land_area_in_sqft', 'bedroomCount', 'bathroomCount', 'area_encoded',
                            'Latitude', 'Longitude']

            # Create a copy to avoid modifying the original DataFrame
            comp_df = reports_data.copy()

            # Convert columns to numeric
            for col in numeric_cols:
                if col in comp_df.columns:
                    comp_df[col] = pd.to_numeric(comp_df[col], errors='coerce')

            # Initialize similarity score
            comp_df['similarity_score'] = 0

            # Property type similarity (highest weight)
            if 'propertyType_encoded' in comp_df.columns:
                comp_df['similarity_score'] += (comp_df['propertyType_encoded'] == property_type_encoded) * 10

            # Property category similarity
            if 'propertyCategory_encoded' in comp_df.columns and property_category:
                comp_df['similarity_score'] += (comp_df['propertyCategory_encoded'] == property_category) * 8

            # Area/location similarity
            if 'area_encoded' in comp_df.columns:
                comp_df['similarity_score'] += (comp_df['area_encoded'] == area_encoded) * 8

            # Size similarity (continuous variable)
            if 'built_up_in_sqft' in comp_df.columns and built_up_area:
                # Calculate percentage difference, capped at 100%
                size_diff_pct = (1 - np.minimum(np.abs(comp_df['built_up_in_sqft'] - built_up_area) /
                                                np.maximum(built_up_area, 1), 1))
                comp_df['similarity_score'] += size_diff_pct * 7

            # Land area similarity for landed properties
            if property_category in [2, 3, 4, 5, 6]:  # Landed property categories
                if 'land_area_in_sqft' in comp_df.columns and land_area:
                    land_diff_pct = (1 - np.minimum(np.abs(comp_df['land_area_in_sqft'] - land_area) /
                                                    np.maximum(land_area, 1), 1))
                    comp_df['similarity_score'] += land_diff_pct * 5

            # Bedroom count similarity
            if 'bedroomCount' in comp_df.columns and bedrooms:
                bedroom_diff = np.abs(comp_df['bedroomCount'] - bedrooms)
                comp_df['similarity_score'] += (bedroom_diff == 0) * 3 + (bedroom_diff == 1) * 1

            # Bathroom count similarity
            if 'bathroomCount' in comp_df.columns and bathrooms:
                bathroom_diff = np.abs(comp_df['bathroomCount'] - bathrooms)
                comp_df['similarity_score'] += (bathroom_diff == 0) * 2 + (bathroom_diff == 1) * 0.5

            # Geographic proximity (if lat/lng available)
            if 'Latitude' in comp_df.columns and 'Longitude' in comp_df.columns and lat and lng:
                # Simple Euclidean distance - could be enhanced with haversine
                dist = np.sqrt(((comp_df['Latitude'] - lat) ** 2) + ((comp_df['Longitude'] - lng) ** 2))
                # Normalize distances to 0-1 range and invert (closer = higher score)
                max_dist = dist.max() if not dist.empty else 1
                if max_dist > 0:
                    proximity_score = (1 - (dist / max_dist))
                    comp_df['similarity_score'] += proximity_score * 5

            # Select top comparable properties (top 20% by similarity or minimum similarity threshold)
            min_similarity = min(comp_df['similarity_score'].quantile(0.80), 15) if len(comp_df) >= 5 else 10
            top_comps = comp_df[comp_df['similarity_score'] >= min_similarity].copy()

            # Calculate price adjustment based on comparables
            if len(top_comps) >= 3:
                # Prioritize more recent listings
                if 'listing_date' in top_comps.columns:
                    top_comps['listing_date'] = pd.to_datetime(top_comps['listing_date'],
                                                                 errors='coerce', dayfirst=True)

                    # Weight by recency and similarity
                    max_date = top_comps['listing_date'].max()
                    if pd.notna(max_date):
                        days_diff = (max_date - top_comps['listing_date']).dt.days
                        max_days = days_diff.max() if not days_diff.empty and days_diff.max() > 0 else 1
                        time_weight = 1 - (days_diff / max_days) * 0.5  # Newer listings get up to 50% more weight

                        # Calculate weighted price per sqft
                        if 'price' in top_comps.columns and 'built_up_in_sqft' in top_comps.columns:
                            top_comps = top_comps.dropna(subset=['price', 'built_up_in_sqft'])
                            top_comps['price_per_sqft'] = top_comps['price'] / top_comps['built_up_in_sqft']

                            # Apply weights (recency + similarity)
                            total_weight = time_weight * top_comps['similarity_score']
                            weighted_ppsf = top_comps['price_per_sqft'] * total_weight

                            # Calculate weighted average price per sqft
                            if total_weight.sum() > 0:
                                weighted_avg_ppsf = weighted_ppsf.sum() / total_weight.sum()

                                # Calculate segment based price
                                segment_price = weighted_avg_ppsf * built_up_area

                                # If segment price differs significantly from model price, adjust
                                if segment_price > 0:
                                    price_ratio = segment_price / base_price

                                    # Adjust segment influence based on confidence (number of comps)
                                    segment_confidence = min(len(top_comps) / 10, 1) * 0.6  # Max 60% weight to comps

                                    if price_ratio > 1.2:  # Comps suggest >20% higher price
                                        segment_adjustment = 1 + (segment_confidence * 0.2)
                                    elif price_ratio < 0.8:  # Comps suggest >20% lower price
                                        segment_adjustment = 1 - (segment_confidence * 0.2)
                                    else:
                                        # Smooth interpolation between bounds
                                        segment_adjustment = 1 + ((price_ratio - 1) * segment_confidence)

        # Apply segmentation adjustment
        base_price *= segment_adjustment

        # 2. MARKET TREND ANALYSIS - Incorporate time-based trends
        trend_adjustment = 1.0

        if not reports_data.empty:
            # Ensure date columns are datetime
            trend_df = reports_data.copy()
            if 'listing_date' in trend_df.columns:
                trend_df['listing_date'] = pd.to_datetime(trend_df['listing_date'],
                                                            errors='coerce', dayfirst=True)

                # Filter out records with invalid dates
                trend_df = trend_df.dropna(subset=['listing_date'])

                if len(trend_df) >= 10:
                    # Create month-year periods
                    trend_df['month_year'] = trend_df['listing_date'].dt.to_period('M')

                    # Calculate monthly median prices
                    monthly_medians = trend_df.groupby('month_year')['price'].median()

                    if len(monthly_medians) >= 3:
                        # Calculate 3-month moving average
                        monthly_medians = monthly_medians.sort_index()
                        moving_avg = monthly_medians.rolling(window=3, min_periods=1).mean()

                        # Get recent trend (last 3 months vs previous 3 months)
                        if len(moving_avg) >= 6:
                            recent_3mo = moving_avg.iloc[-3:].mean()
                            previous_3mo = moving_avg.iloc[-6:-3].mean()

                            if previous_3mo > 0:
                                trend_pct = (recent_3mo / previous_3mo) - 1

                                # Apply trend but cap the impact (max ±8%)
                                trend_adjustment = 1 + np.clip(trend_pct, -0.08, 0.08)

        # Apply trend adjustment
        base_price *= trend_adjustment

        # 4. PROPERTY CONDITION PREMIUM/DISCOUNT
        condition_adjustment = 1.0

        # Get condition values (scaled 0-4, with 2 being standard)
        kitchen_val = int(input_data.get('kitchen', 2))
        bathroom_val = int(input_data.get('bathroom', 2))
        living_room_val = int(input_data.get('livingRoom', 2))
        home_exterior_val = int(input_data.get('homeExterior', 2))

        # Calculate weighted condition score (kitchen has highest impact)
        condition_score = (
                kitchen_val * 0.4 +  # Kitchen (highest weight)
                bathroom_val * 0.25 +  # Bathroom
                living_room_val * 0.2 +  # Living room
                home_exterior_val * 0.15  # Exterior
        )

        # Calculate adjustment (each point above/below standard is ±5%)
        condition_modifier = (condition_score - 2) * 0.05
        condition_adjustment = 1 + condition_modifier

        # Apply condition adjustment
        base_price *= condition_adjustment

        # 6. PREMIUM FEATURES & NEGATIVE FACTORS
        feature_adjustment = 1.0
        premium_features = 0
        negative_features = 0

        # Check for premium features
        if input_data.get('private_pool', 0) == 1:
            premium_features += 0.03  # Private pool: +3%

        if input_data.get('solar', 0) == 1:
            premium_features += 0.02  # Solar panels: +2%

        if input_data.get('lift', 0) == 1:
            premium_features += 0.02  # Private lift: +2%

        if input_data.get('gated_guarded', 0) == 1:
            premium_features += 0.02  # Gated & guarded: +2%

        if input_data.get('1.5_km_within_mrt', 0) == 1:
            premium_features += 0.03  # Near MRT: +3%

        # Check for negative factors (foundation issues, bad views, etc.)
        if input_data.get('foundation_issues', 0) == 1:
            negative_features += 0.08  # Foundation issues: -8%

        # Check view factor based on property type
        property_type_1 = input_data.get('propertyType$1')
        view_val = int(input_data.get('view', 4))

        if property_type_1 == '0':  # Landed property
            if view_val == 0:  # Facing T-junction
                negative_features += 0.03
            elif view_val == 1:  # Facing busy road
                negative_features += 0.02
            elif view_val == 2:  # Facing park
                premium_features += 0.02
            elif view_val == 3:  # Home entrance lower than road
                negative_features += 0.02
        elif property_type_1 == '1':  # High-rise property
            if view_val == 0:  # City view
                premium_features += 0.01
            elif view_val == 1:  # Forest/Greens
                premium_features += 0.02
            elif view_val == 2:  # Unobstructed
                premium_features += 0.02
            elif view_val == 3:  # Landmark view
                premium_features += 0.04
            elif view_val == 4:  # Cemetery
                negative_features += 0.05

        # Calculate net feature adjustment
        feature_adjustment = 1 + premium_features - negative_features

        # Apply feature adjustment
        base_price *= feature_adjustment

        # 9. SANITY CHECKS & FINAL ADJUSTMENTS
        # Ensure the price doesn't deviate too much from base prediction
        price_floor = float(predicted_price) * 0.8
        price_ceiling = float(predicted_price) * 1.2

        # Apply bounds
        final_price = max(min(base_price, price_ceiling), price_floor)

        # Convert to cents and round to nearest thousand
        price_cents = int(final_price * 100)
        from price_prediction import PricePredictor
        final_price_cents = PricePredictor.round_cents_to_thousand(price_cents)

        return final_price_cents
    @staticmethod
    def calculate_estimated_sold_price(reports_data: pd.DataFrame, predicted_price: float) -> int:
        """
        Calculate the estimated sold price based on market trends.

        Args:
            reports_data: Property data
            predicted_price: Predicted price from model

        Returns:
            int: Estimated sold price in cents
        """
        # Start with the base predicted price
        base_price = float(predicted_price)

        # Apply a simple market-based adjustment if we have data with both listing and sale prices
        if not reports_data.empty and 'price' in reports_data.columns:
            if 'original_price' in reports_data.columns:
                # Calculate average price change
                reports_data['price_change'] = (reports_data['price'] - reports_data['original_price']) / reports_data['original_price']
                avg_price_change = reports_data['price_change'].mean()

                if not pd.isna(avg_price_change):
                    # Apply the adjustment
                    adjustment = max(min(avg_price_change, 0.1), -0.1)  # Cap at ±10%
                    base_price = base_price * (1 + adjustment)

        # Convert to cents and round
        price_cents = int(base_price * 100)
        from price_prediction import PricePredictor
        final_price_cents = PricePredictor.round_cents_to_thousand(price_cents)

        return final_price_cents
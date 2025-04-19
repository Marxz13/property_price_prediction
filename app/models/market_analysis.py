"""
Market analysis utilities for property valuation.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from app.models.property_types import PropertyTypeClassifier, PropertyTypeMappings
from config import config


class MarketAnalyzer:
    """Class for analyzing property market data."""

    def __init__(self):
        """Initialize market analyzer with database connections."""
        self.db_client = config.get_mongodb_client()

    def get_transaction_data(self, limit=1000) -> pd.DataFrame:
        """
        Retrieve property transaction data from MongoDB.

        Args:
            limit: Maximum number of records to retrieve

        Returns:
            DataFrame: Property transaction data
        """
        collection = self.db_client["reports"]["transacted"]
        cursor = collection.find().limit(limit)
        return pd.DataFrame(list(cursor))

    def get_active_listings(self, limit=1000) -> pd.DataFrame:
        """
        Retrieve active property listings from MongoDB.

        Args:
            limit: Maximum number of records to retrieve

        Returns:
            DataFrame: Active property listings
        """
        collection = self.db_client["reports"]["active"]
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

    def calculate_hierarchical_median_price(self, mongodb_data: pd.DataFrame,
                                            input_data: Dict,
                                            postal_code: int,
                                            sublocality: str,
                                            locality: str,
                                            time_filter: Optional[pd.Series] = None) -> Tuple[float, int]:
        """
        Calculate median sale price using hierarchical matching approach.

        Args:
            mongodb_data: Historical property data
            input_data: Current property details
            postal_code: Property postal code
            sublocality: Property sublocality
            locality: Property locality/city
            time_filter: Boolean mask for time period filtering

        Returns:
            Tuple[float, int]: Median sale price and number of records used
        """
        # Make a copy to avoid modifying the original
        data = mongodb_data.copy()

        # Apply time filter if provided
        if time_filter is not None:
            data = data[time_filter].copy()

        if data.empty:
            return 0, 0

        # Get property details
        try:
            property_type_encoded = PropertyTypeClassifier.map_property_type(input_data)
            property_category = PropertyTypeClassifier.get_property_category(input_data)
        except Exception:
            property_type_encoded = None
            property_category = None

        # Ensure numeric columns for filtering
        for col in ['propertyType_encoded', 'propertyCategory_encoded']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Level 1: postal_code + sublocality + exact property type
        if property_type_encoded is not None and sublocality is not None:
            filtered = data[
                (data['postal_code'] == postal_code) &
                (data['area'] == sublocality) &
                (data['propertyType_encoded'] == property_type_encoded)
                ]

            if len(filtered) >= 5:
                median = self.calc_median_price(pd.to_numeric(filtered['sold_price'], errors='coerce'))
                return median, len(filtered)

        # Level 2: postal_code + sublocality + property category
        if property_category is not None and sublocality is not None:
            filtered = data[
                (data['postal_code'] == postal_code) &
                (data['area'] == sublocality) &
                (data['propertyCategory_encoded'] == property_category)
                ]

            if len(filtered) >= 5:
                median = self.calc_median_price(pd.to_numeric(filtered['sold_price'], errors='coerce'))
                return median, len(filtered)

        # Level 3: postal_code + locality + exact property type
        if property_type_encoded is not None and locality is not None:
            locality_filter = (
                    data['city'].str.contains(locality, case=False, na=False) |
                    data['township'].str.contains(locality, case=False, na=False)
            )
            filtered = data[
                (data['postal_code'] == postal_code) &
                locality_filter &
                (data['propertyType_encoded'] == property_type_encoded)
                ]

            if len(filtered) >= 5:
                median = self.calc_median_price(pd.to_numeric(filtered['sold_price'], errors='coerce'))
                return median, len(filtered)

        # Level 4: postal_code + locality + property category
        if property_category is not None and locality is not None:
            locality_filter = (
                    data['city'].str.contains(locality, case=False, na=False) |
                    data['township'].str.contains(locality, case=False, na=False)
            )
            filtered = data[
                (data['postal_code'] == postal_code) &
                locality_filter &
                (data['propertyCategory_encoded'] == property_category)
                ]

            if len(filtered) >= 5:
                median = self.calc_median_price(pd.to_numeric(filtered['sold_price'], errors='coerce'))
                return median, len(filtered)

        # Fallback: postal_code + sublocality (all property types)
        if sublocality is not None:
            filtered = data[
                (data['postal_code'] == postal_code) &
                (data['area'] == sublocality)
                ]

            if len(filtered) >= 3:  # Lower threshold for fallback
                median = self.calc_median_price(pd.to_numeric(filtered['sold_price'], errors='coerce'))
                return median, len(filtered)

        # Final fallback: just postal_code
        filtered = data[data['postal_code'] == postal_code]

        if len(filtered) >= 3:  # Lower threshold for last resort
            median = self.calc_median_price(pd.to_numeric(filtered['sold_price'], errors='coerce'))
            return median, len(filtered)

        # If all else fails
        return 0, 0

    @staticmethod
    def calc_avg_days_on_market(total_days: float, num_sold: int) -> float:
        """
        Calculate average days on market.

        Args:
            total_days: Sum of days on market
            num_sold: Number of properties sold

        Returns:
            float: Average days on market
        """
        return 0 if num_sold == 0 else total_days / num_sold

    def calc_avg_days_on_market_over_time(self, mongodb_data: pd.DataFrame,
                                          input_data: Optional[Dict] = None,
                                          sublocality: Optional[str] = None,
                                          locality: Optional[str] = None) -> List[Dict]:
        """
        Calculate average days on market over time, filtered by property category.
        Forward-fills missing months with the last known value.

        Args:
            mongodb_data: Historical data
            input_data: Input property data for category matching
            sublocality: Sublocality to filter by
            locality: Locality to filter by

        Returns:
            List[Dict]: List of dictionaries with date and days
        """
        # Clone the data to avoid modifying the original
        data = mongodb_data.copy()

        # Ensure datetime format
        data['date_of_asking_price'] = pd.to_datetime(
            data['date_of_asking_price'], dayfirst=True, errors='coerce'
        )
        data['date_of_sold_price'] = pd.to_datetime(
            data['date_of_sold_price'], dayfirst=True, errors='coerce'
        )

        # Calculate days on market
        data['days_on_market'] = (
                data['date_of_sold_price'] - data['date_of_asking_price']
        ).dt.days

        # Filter out invalid days on market
        valid_data = data.dropna(subset=['date_of_asking_price', 'date_of_sold_price', 'days_on_market']).copy()
        valid_data = valid_data[valid_data['days_on_market'] >= 0]

        # Filter to only include records from the last 6 months
        six_months_ago = pd.Timestamp.today() - pd.DateOffset(months=6)
        valid_data = valid_data[valid_data['date_of_asking_price'] >= six_months_ago]

        if valid_data.empty:
            return []

        # Apply property category filtering if input_data is provided
        if input_data is not None:
            # Determine if high-rise or landed
            pt1 = input_data.get("propertyType$1")
            is_highrise = pt1 == "1"

            # Get property category
            property_str = PropertyTypeClassifier.get_property_string_from_code(input_data)
            input_property_category = None

            if property_str:
                input_property_category = PropertyTypeMappings.CATEGORY_MAP.get(property_str.lower())

            # Apply hierarchical filtering
            filtered_data = None

            # 1. Exact category + Sublocality
            if input_property_category is not None and sublocality is not None:
                category_filter = valid_data['propertyCategory_encoded'] == float(input_property_category)
                sublocality_filter = (
                        valid_data['area'].str.contains(sublocality, case=False, na=False) |
                        valid_data['township'].str.contains(sublocality, case=False, na=False)
                )
                filtered_data = valid_data[category_filter & sublocality_filter]

            # 2. Similar categories + Sublocality
            if filtered_data is None or len(filtered_data) < 10:
                if is_highrise:
                    highrise_categories = [1.0, 7.0, 8.0, 9.0, 10.0, 11.0]
                    category_filter = valid_data['propertyCategory_encoded'].isin(highrise_categories)
                else:
                    landed_categories = [2.0, 3.0, 4.0, 5.0, 6.0]
                    category_filter = valid_data['propertyCategory_encoded'].isin(landed_categories)

                if sublocality is not None:
                    sublocality_filter = (
                            valid_data['area'].str.contains(sublocality, case=False, na=False) |
                            valid_data['township'].str.contains(sublocality, case=False, na=False)
                    )
                    similar_category_data = valid_data[category_filter & sublocality_filter]

                    if filtered_data is None:
                        filtered_data = similar_category_data
                    elif len(filtered_data) < 10:
                        # Combine with previous data, avoiding duplicates
                        existing_indices = set(filtered_data.index)
                        new_records = similar_category_data[~similar_category_data.index.isin(existing_indices)]
                        filtered_data = pd.concat([filtered_data, new_records])

            # 3. Exact category + Locality
            if filtered_data is None or len(filtered_data) < 10:
                if input_property_category is not None and locality is not None:
                    category_filter = valid_data['propertyCategory_encoded'] == float(input_property_category)
                    locality_filter = (
                            valid_data['area'].str.contains(locality, case=False, na=False) |
                            valid_data['township'].str.contains(locality, case=False, na=False) |
                            valid_data['city'].str.contains(locality, case=False, na=False)
                    )
                    locality_category_data = valid_data[category_filter & locality_filter]

                    if filtered_data is None:
                        filtered_data = locality_category_data
                    elif len(filtered_data) < 10:
                        # Combine with previous data, avoiding duplicates
                        existing_indices = set(filtered_data.index)
                        new_records = locality_category_data[~locality_category_data.index.isin(existing_indices)]
                        filtered_data = pd.concat([filtered_data, new_records])

            # 4. Similar categories + Locality
            if filtered_data is None or len(filtered_data) < 10:
                if locality is not None:
                    if is_highrise:
                        highrise_categories = [1.0, 7.0, 8.0, 9.0, 10.0, 11.0]
                        category_filter = valid_data['propertyCategory_encoded'].isin(highrise_categories)
                    else:
                        landed_categories = [2.0, 3.0, 4.0, 5.0, 6.0]
                        category_filter = valid_data['propertyCategory_encoded'].isin(landed_categories)

                    locality_filter = (
                            valid_data['area'].str.contains(locality, case=False, na=False) |
                            valid_data['township'].str.contains(locality, case=False, na=False) |
                            valid_data['city'].str.contains(locality, case=False, na=False)
                    )
                    similar_locality_data = valid_data[category_filter & locality_filter]

                    if filtered_data is None:
                        filtered_data = similar_locality_data
                    elif len(filtered_data) < 10:
                        # Combine with previous data, avoiding duplicates
                        existing_indices = set(filtered_data.index)
                        new_records = similar_locality_data[~similar_locality_data.index.isin(existing_indices)]
                        filtered_data = pd.concat([filtered_data, new_records])

            # Use filtered data if we have enough, otherwise fallback to original valid data
            if filtered_data is not None and len(filtered_data) >= 5:
                valid_data = filtered_data

        # Group by month and calculate average days on market
        valid_data['month'] = valid_data['date_of_asking_price'].dt.to_period('M')
        avg_days_on_market = valid_data.groupby('month')['days_on_market'].mean().reset_index()

        if len(avg_days_on_market) == 0:
            return []

        # Generate a complete sequence of months for the past 6 months
        end_month = pd.Timestamp.today().to_period('M')
        start_month = (pd.Timestamp.today() - pd.DateOffset(months=5)).to_period('M')
        complete_months = pd.DataFrame({'month': pd.period_range(start=start_month, end=end_month, freq='M')})

        # Merge with actual data
        merged_data = pd.merge(complete_months, avg_days_on_market, on='month', how='left')

        # Forward fill missing values (last known value carried forward)
        merged_data['days_on_market'] = merged_data['days_on_market'].ffill()

        # Format the result
        merged_data['date'] = merged_data['month'].apply(lambda x: x.start_time.isoformat())
        merged_data['days'] = merged_data['days_on_market'].apply(
            lambda d: max(round(d), 0) if pd.notna(d) else 0
        ).astype(int)

        return merged_data[['date', 'days']].to_dict(orient="records")

    @staticmethod
    def calc_median_trend(one_month_median: float, three_months_median: float) -> str:
        """
        Calculate trend direction based on median prices.

        Args:
            one_month_median: Recent median price
            three_months_median: Previous median price

        Returns:
            str: Trend code ("0"=down, "1"=stable, "2"=up)
        """
        if one_month_median is None or three_months_median is None:
            return "1"  # Default: stable

        try:
            one_month_median = float(one_month_median)
            three_months_median = float(three_months_median)
        except ValueError:
            return "1"

        if one_month_median > three_months_median:
            return "2"  # Up
        elif one_month_median < three_months_median:
            return "0"  # Down
        else:
            return "1"  # Stable

    @staticmethod
    def predict_trend(value_1: float, value_2: float, trend_type: Optional[str] = None) -> str:
        """
        Predict trend direction based on two values.

        Args:
            value_1: Recent value
            value_2: Previous value
            trend_type: Type of trend (specialized handling for days on market)

        Returns:
            str: Trend code ("0"=down, "1"=stable, "2"=up)
        """
        if value_1 is None or value_2 is None:
            return "1"

        try:
            value_1 = float(value_1)
            value_2 = float(value_2)
        except ValueError:
            return "1"

        if value_1 > value_2:
            # For days on market, increase is negative
            return "2" if trend_type != "avg_days_on_market" else "0"
        elif value_1 < value_2:
            # For days on market, decrease is positive
            return "0" if trend_type != "avg_days_on_market" else "2"
        else:
            return "1"  # Stable

    def calculate_estimated_sold_price(self, df: pd.DataFrame, predicted_price: float) -> int:
        """
        Calculate the estimated sold price based on market trends.

        Args:
            df: Historical sales data
            predicted_price: Predicted price from model

        Returns:
            int: Estimated sold price in cents
        """
        df = df[df['sold_price'] > 0]

        if df.empty:
            return int(float(predicted_price) * 100)

        df['date_diff'] = (df['date_of_sold_price'] - df['date_of_asking_price']).dt.days
        df['date_diff'] = df['date_diff'].replace(0, 1)
        df['monthly_price_change'] = ((df['sold_price'] - df['asking_price']) / df['asking_price']) / (
                    df['date_diff'] / 30)

        market_rate_avg = df['monthly_price_change'].mean()
        if pd.isna(market_rate_avg):
            market_rate_avg = 0

        # Apply the market rate to the predicted price
        if market_rate_avg < 0:
            depreciation_rate = abs(market_rate_avg)
            return int(float(predicted_price) * (1 - depreciation_rate * 3) * 100)
        else:
            growth_rate = market_rate_avg
            return int(float(predicted_price) * (1 + growth_rate * 3) * 100)

    def calculate_recommended_selling_price(self, df: pd.DataFrame,
                                            predicted_price: float,
                                            input_data: Dict,
                                            active_df: Optional[pd.DataFrame] = None) -> int:
        """
        Calculate an enhanced recommended selling price using multiple factors.

        Args:
            df: Historical property sales data
            predicted_price: Base predicted price from the ML model
            input_data: Property details and features
            active_df: Current active listings in the market

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

        if not df.empty:
            # Convert numeric columns
            numeric_cols = ['propertyType_encoded', 'propertyCategory_encoded', 'built_up_in_sqft',
                            'land_area_in_sqft', 'bedroomCount', 'bathroomCount', 'area_encoded',
                            'Latitude', 'Longitude']

            # Create a copy to avoid modifying the original DataFrame
            comp_df = df.copy()

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
                # Prioritize more recent sales
                top_comps['date_of_sold_price'] = pd.to_datetime(top_comps['date_of_sold_price'],
                                                                 errors='coerce', dayfirst=True)

                # Weight by recency and similarity
                max_date = top_comps['date_of_sold_price'].max()
                if pd.notna(max_date):
                    days_diff = (max_date - top_comps['date_of_sold_price']).dt.days
                    max_days = days_diff.max() if not days_diff.empty and days_diff.max() > 0 else 1
                    time_weight = 1 - (days_diff / max_days) * 0.5  # Newer sales get up to 50% more weight

                    # Calculate weighted price per sqft
                    if 'sold_price' in top_comps.columns and 'built_up_in_sqft' in top_comps.columns:
                        top_comps = top_comps.dropna(subset=['sold_price', 'built_up_in_sqft'])
                        top_comps['price_per_sqft'] = top_comps['sold_price'] / top_comps['built_up_in_sqft']

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

        if not df.empty:
            # Ensure date columns are datetime
            trend_df = df.copy()
            trend_df['date_of_sold_price'] = pd.to_datetime(trend_df['date_of_sold_price'],
                                                            errors='coerce', dayfirst=True)

            # Filter out records with invalid dates
            trend_df = trend_df.dropna(subset=['date_of_sold_price'])

            if len(trend_df) >= 10:
                # Create month-year periods
                trend_df['month_year'] = trend_df['date_of_sold_price'].dt.to_period('M')

                # Calculate monthly median prices
                monthly_medians = trend_df.groupby('month_year')['sold_price'].median()

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

        # 5. MARKET ACTIVITY & COMPETITION
        competition_adjustment = 1.0

        if active_df is not None and not active_df.empty:
            comp_analysis = active_df.copy()

            # Filter to relevant area
            area_filter = False
            for component in input_data.get('place', {}).get('addressComponents', []):
                if 'locality' in component.get('types', []):
                    area_value = component.get('longText', '')
                    if area_value and len(area_value) > 3:
                        area_filter = (
                            comp_analysis['area'].str.contains(area_value, case=False, na=False)
                        )
                        break

            if isinstance(area_filter, pd.Series) and not area_filter.empty:
                local_market = comp_analysis[area_filter].copy()

                # Further filter by property type if possible
                if 'propertyType_encoded' in local_market.columns:
                    local_market['propertyType_encoded'] = pd.to_numeric(
                        local_market['propertyType_encoded'], errors='coerce')

                    # Get similar property types
                    if property_type_1 == '0':  # Landed
                        similar_types = local_market[
                            (local_market['propertyType_encoded'] >= 2) &
                            (local_market['propertyType_encoded'] <= 17)
                            ]
                    else:  # Highrise
                        similar_types = local_market[
                            (local_market['propertyType_encoded'] == 1) |
                            (local_market['propertyType_encoded'] >= 18)
                            ]

                    if len(similar_types) >= 5:
                        local_market = similar_types

                if len(local_market) >= 5:
                    num_competing = len(local_market)

                    # Median days on market for active listings
                    if 'date_of_asking_price' in local_market.columns:
                        local_market['date_of_asking_price'] = pd.to_datetime(
                            local_market['date_of_asking_price'], errors='coerce', dayfirst=True)
                        local_market['days_on_market'] = (
                                datetime.now() - local_market['date_of_asking_price']).dt.days
                        median_dom = local_market['days_on_market'].median()

                        # More competition and longer days on market suggest price pressure
                        if num_competing > 20 and median_dom > 90:
                            competition_adjustment = 0.95  # High competition, slow market: -5%
                        elif num_competing > 20:
                            competition_adjustment = 0.97  # High competition: -3%
                        elif median_dom > 90:
                            competition_adjustment = 0.98  # Slow market: -2%
                        elif num_competing < 5 and median_dom < 45:
                            competition_adjustment = 1.03  # Low competition, fast market: +3%

        # Apply competition adjustment
        base_price *= competition_adjustment

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
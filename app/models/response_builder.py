"""
Response generation for property valuation API.
"""
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from datetime import datetime

from app.models.property_types import PropertyTypeClassifier
from app.models.price_prediction import PricePredictor
from app.models.market_analysis import MarketAnalyzer


class PropertyComparator:
    """Get comparable properties for analysis."""

    def __init__(self, db_client):
        """
        Initialize with database client.

        Args:
            db_client: MongoDB client
        """
        self.db_client = db_client

    def get_last_5_sold_properties(self, input_data: Dict, area_filter: str,
                                   input_postal_code, payload_property_type) -> List[Dict]:
        """
        Find comparable sold properties based on input property details.

        Args:
            input_data: Property data dictionary
            area_filter: Area/sublocality to filter by
            input_postal_code: Postal code to filter by
            payload_property_type: Property type code

        Returns:
            List[Dict]: List of comparable property details
        """
        try:
            # Sanitize inputs
            if input_postal_code is not None:
                input_postal_code = str(input_postal_code)
            else:
                input_postal_code = ""

            # Get the built-up area from input_data
            input_built_up = input_data.get("propertyDetails$1", {}).get("buildUpAreaInSqft", 0)
            if input_built_up:
                try:
                    input_built_up = float(input_built_up)
                except (ValueError, TypeError):
                    input_built_up = 0

            # Extract locality information with more detail
            locality = None
            sublocality_level_1 = None
            for comp in input_data.get("place", {}).get("addressComponents", []):
                if "locality" in comp.get("types", []):
                    locality = comp.get("longText")
                if "sublocality_level_1" in comp.get("types", []):
                    sublocality_level_1 = comp.get("longText")

            # Use the most specific location info available
            primary_location = sublocality_level_1 or area_filter or locality or ""

            # Get the property type from input data
            pt1 = input_data.get("propertyType$1")
            is_highrise = pt1 == "1"

            # Get exact property type encoded for better matching
            try:
                exact_property_type_encoded = PropertyTypeClassifier.map_property_type(input_data)
            except Exception:
                exact_property_type_encoded = None

            # Determine property category based on the property type
            input_property_category = None
            property_str = PropertyTypeClassifier.get_property_string_from_code(input_data)

            if property_str:
                property_str_lower = property_str.lower()
                from property_types import PropertyTypeMappings
                input_property_category = PropertyTypeMappings.CATEGORY_MAP.get(property_str_lower)

            # Set up MongoDB connection
            try:
                sold_db = self.db_client["reports"]
                sold_collection = sold_db["transacted"]

                # Define filters for exact property type if available
                exact_type_filter = None
                if exact_property_type_encoded is not None:
                    exact_type_filter = {"propertyType_encoded": float(exact_property_type_encoded)}

                # Define category filters
                if is_highrise:
                    if input_property_category is not None:
                        exact_category_filter = {"propertyCategory_encoded": float(input_property_category)}
                        similar_category_filter = {
                            "propertyCategory_encoded": {"$in": [1.0, 7.0, 8.0, 9.0, 10.0, 11.0]}}
                    else:
                        exact_category_filter = None
                        similar_category_filter = {
                            "propertyCategory_encoded": {"$in": [1.0, 7.0, 8.0, 9.0, 10.0, 11.0]}}
                else:
                    if input_property_category is not None:
                        exact_category_filter = {"propertyCategory_encoded": float(input_property_category)}
                        similar_category_filter = {"propertyCategory_encoded": {"$in": [2.0, 3.0, 4.0, 5.0, 6.0]}}
                    else:
                        exact_category_filter = None
                        similar_category_filter = {"propertyCategory_encoded": {"$in": [2.0, 3.0, 4.0, 5.0, 6.0]}}

                # Try multiple matching approaches in priority order
                category_matches = []

                # 1. First try: Exact property type if available
                if exact_type_filter:
                    cursor = sold_collection.find(exact_type_filter)
                    type_matches = list(cursor)
                    if len(type_matches) >= 10:
                        category_matches = type_matches

                # 2. If not enough matches, try exact category
                if len(category_matches) < 10 and exact_category_filter:
                    cursor = sold_collection.find(exact_category_filter)
                    cat_matches = list(cursor)
                    if not category_matches:
                        category_matches = cat_matches
                    elif len(cat_matches) > 0:
                        # Combine matches, avoiding duplicates
                        existing_ids = {m.get("_id") for m in category_matches if "_id" in m}
                        new_matches = [m for m in cat_matches if m.get("_id") not in existing_ids]
                        category_matches.extend(new_matches)

                # 3. If still not enough, try similar categories
                if len(category_matches) < 10:
                    cursor = sold_collection.find(similar_category_filter)
                    similar_matches = list(cursor)
                    if not category_matches:
                        category_matches = similar_matches
                    elif len(similar_matches) > 0:
                        # Combine matches, avoiding duplicates
                        existing_ids = {m.get("_id") for m in category_matches if "_id" in m}
                        new_matches = [m for m in similar_matches if m.get("_id") not in existing_ids]
                        category_matches.extend(new_matches)

                if not category_matches:
                    return []

                # Convert to DataFrame for easier processing
                df = pd.DataFrame(category_matches).copy()

                # Apply enhanced fuzzy matching on location
                try:
                    from fuzzywuzzy import fuzz

                    # Function to calculate fuzzy match score with higher weights for township
                    def calculate_location_score(row, target):
                        if pd.isna(row['township']) and pd.isna(row['area']):
                            return 0

                        # Township match has higher priority (weight=1.2)
                        township_score = fuzz.token_sort_ratio(str(row['township']).lower(),
                                                               target.lower()) * 1.2 if not pd.isna(
                            row['township']) else 0

                        # Area match has standard priority (weight=1.0)
                        area_score = fuzz.token_sort_ratio(str(row['area']).lower(), target.lower()) if not pd.isna(
                            row['area']) else 0

                        # Return the higher of the weighted scores, capped at 100
                        return min(max(township_score, area_score), 100)

                    # Add fuzzy match score column with priority on township/sublocality
                    if primary_location:
                        df['location_score'] = df.apply(lambda row: calculate_location_score(row, primary_location),
                                                        axis=1)

                        # Find high-quality matches (90%+ score)
                        high_matches = df[df['location_score'] >= 90].copy()

                        if len(high_matches) >= 5:
                            # If we have enough high matches, use only those
                            df = high_matches
                        else:
                            # Try with a lower threshold
                            good_matches = df[df['location_score'] >= 80].copy()

                            if len(good_matches) >= 5:
                                df = good_matches

                except ImportError:
                    # Fallback to exact matching
                    if primary_location:
                        location_filter = (
                                df['township'].str.contains(primary_location, case=False, na=False) |
                                df['area'].str.contains(primary_location, case=False, na=False)
                        )
                        location_matches = df[location_filter].copy()

                        if len(location_matches) >= 5:
                            df = location_matches

                # If we still don't have enough matches, take top properties by date
                if len(df) < 5:
                    df = pd.DataFrame(category_matches).sort_values(
                        by='date_of_sold_price', ascending=False
                    ).head(10).copy()

                # Calculate similarity score based on built_up_in_sqft
                if input_built_up > 0 and 'built_up_in_sqft' in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    df['built_up_in_sqft'] = pd.to_numeric(df['built_up_in_sqft'], errors='coerce')

                    # Calculate the relative difference in built up area
                    df['built_up_diff_pct'] = abs(df['built_up_in_sqft'] - input_built_up) / max(input_built_up,
                                                                                                 1) * 100

                    # Sort by similarity score (most similar first)
                    df_sorted = df.sort_values(by=['built_up_diff_pct'])

                    # Take top 5 most similar properties
                    last_5 = df_sorted.head(5).copy()
                else:
                    # If no built_up area to compare, sort by date
                    if 'date_of_sold_price' in df.columns:
                        df = df.sort_values(by='date_of_sold_price', ascending=False)
                    last_5 = df.head(5).copy()

                # Ensure all required fields are present
                essential_fields = ['house_no', 'street_no', 'township', 'area', 'state', 'country',
                                    'price_per_sqft', 'sold_price', 'date_of_sold_price']
                missing_fields = [field for field in essential_fields if field not in last_5.columns]
                if missing_fields:
                    for field in missing_fields:
                        last_5[field] = "N/A"

                # Format dates
                last_5['date_of_sold_price'] = pd.to_datetime(last_5['date_of_sold_price'],
                                                              dayfirst=True, errors='coerce')

                # Fill missing values
                if 'land_area_in_sqft' not in last_5.columns:
                    last_5['land_area_in_sqft'] = 0
                if 'built_up_in_sqft' not in last_5.columns:
                    last_5['built_up_in_sqft'] = 0

                properties_list = []

                # Abbreviations for formatting
                area_abbreviations = {
                    "Mont Kiara": "MK", "Petaling Jaya": "PJ",
                    "Subang Jaya": "SJ", "Taman Tun Dr Ismail": "TTDI",
                    "Damansara": "DJ"
                }
                country_abbreviations = {"Malaysia": "MY"}
                state_abbreviations = {"Selangor": "SGR", "Kuala Lumpur": "KL"}

                # Get floors from input
                from feature_engineering import FeatureEngineer
                input_floors = FeatureEngineer.extract_floors_from_input(input_data)

                # Determine if property is landed
                try:
                    property_type_num = int(float(payload_property_type))
                    is_landed = 2 <= property_type_num <= 17
                except (ValueError, TypeError):
                    is_landed = False  # Default

                for _, row in last_5.iterrows():
                    try:
                        # Format address components
                        house_no = str(row.get('house_no', 'N/A'))
                        street_no = str(row.get('street_no', 'N/A'))
                        township = str(row.get('township', 'N/A'))
                        area = str(row.get('area', 'N/A'))
                        state = str(row.get('state', 'N/A'))
                        country = str(row.get('country', 'Malaysia'))

                        short_area = area_abbreviations.get(area, area)
                        short_country = country_abbreviations.get(country, country)
                        short_state = state_abbreviations.get(state, state)

                        address = [
                            {"longText": house_no, "shortText": house_no, "types": ["house_no"]},
                            {"longText": street_no, "shortText": street_no, "types": ["street_no"]},
                            {"longText": township, "shortText": township, "types": ["township"]},
                            {"longText": area, "shortText": short_area, "types": ["area"]},
                            {"longText": state, "shortText": short_state, "types": ["state"]},
                            {"longText": country, "shortText": short_country, "types": ["country"]},
                            {"longText": input_postal_code, "shortText": input_postal_code, "types": ["postal_code"]},
                        ]

                        # Set floors value
                        if is_landed:
                            floors = str(input_floors)
                        else:
                            original_floors = row.get('floors')
                            floors = str(original_floors) if pd.notna(original_floors) else "1"

                        # Get land area as string per schema requirement
                        land_area = row.get('land_area_in_sqft', 0)
                        try:
                            land_area_int = int(float(land_area)) if pd.notna(land_area) else 0
                        except (ValueError, TypeError):
                            land_area_int = 0

                        # Get built-up area
                        built_up = row.get('built_up_in_sqft', 0)
                        try:
                            built_up_int = int(float(built_up)) if pd.notna(built_up) else 0
                        except (ValueError, TypeError):
                            built_up_int = 0

                        # Get price per sqft
                        price_per_sqft = row.get('price_per_sqft', 0)
                        try:
                            price_per_sqft_int = int(float(price_per_sqft) * 100) if pd.notna(price_per_sqft) else 0
                        except (ValueError, TypeError):
                            price_per_sqft_int = 0

                        # Get sold price
                        sold_price = row.get('sold_price', 0)
                        try:
                            sold_price_int = int(float(sold_price) * 100) if pd.notna(sold_price) else 0
                        except (ValueError, TypeError):
                            sold_price_int = 0

                        # Get tenure
                        tenure = row.get('tenure_encoded', '1')
                        tenure = str(tenure) if pd.notna(tenure) else '1'

                        # Create property details object with proper schema compliance
                        property_details = {
                            "address": address,
                            "type": str(payload_property_type),
                            "tenure": tenure,
                            "floors": floors,
                            "rooms": 1,
                            "land_area_in_sqft": land_area_int,
                            "built_up_in_sqft": built_up_int,
                            "price_per_sqft": price_per_sqft_int,
                            "price": sold_price_int
                        }
                        properties_list.append(property_details)

                    except Exception as e:
                        print(f"Error processing property row: {e}")
                        continue

                return properties_list

            except Exception as e:
                print(f"Error accessing MongoDB: {e}")
                return []

        except Exception:
            return []


class ResponseBuilder:
    """Build response for property valuation API."""

    def __init__(self, db_client, price_predictor: PricePredictor, market_analyzer: MarketAnalyzer):
        """
        Initialize with necessary components.

        Args:
            db_client: MongoDB client
            price_predictor: Instance of PricePredictor class
            market_analyzer: Instance of MarketAnalyzer class
        """
        self.db_client = db_client
        self.price_predictor = price_predictor
        self.market_analyzer = market_analyzer
        self.property_comparator = PropertyComparator(db_client)

    def get_response_json(self, df: pd.DataFrame,
                          predicted_result: float,
                          input_address_components: List[Dict],
                          input_data: Dict) -> Dict:
        """
        Generate comprehensive market analysis response.

        Args:
            df: Historical sales data
            predicted_result: Predicted price from model
            input_address_components: Address components of the subject property
            input_data: Property details and features

        Returns:
            Dict: Complete market analysis response
        """
        # ---------------------------
        # Active Market Metrics from the "active" Collection
        # ---------------------------
        active_collection = self.db_client["reports"]["active"]
        active_data = list(active_collection.find({}))
        active_df = pd.DataFrame(active_data)
        active_df['date_of_asking_price'] = pd.to_datetime(
            active_df['date_of_asking_price'],
            format='%d-%b-%y',
            errors='coerce'
        )
        active_df['postal_code'] = active_df['postal_code'].astype(int)

        # ---------------------------
        # Compute date boundaries for fallback windows
        # ---------------------------
        one_month_ago = pd.Timestamp.today() - pd.DateOffset(months=1)
        three_months_ago = pd.Timestamp.today() - pd.DateOffset(months=3)
        six_months_ago = pd.Timestamp.today() - pd.DateOffset(months=6)

        # ---------------------------
        # Extract postal_code and sublocality from input_address_components
        # ---------------------------
        postal_code = next(
            (c.get("longText") for c in input_address_components if "postal_code" in c.get("types", [])),
            None
        )
        if postal_code is not None:
            postal_code = int(postal_code)
        sublocality = next(
            (c.get("longText") for c in input_address_components if "sublocality_level_1" in c.get("types", [])),
            None
        )
        locality = next(
            (c.get("longText") for c in input_address_components if "locality" in c.get("types", [])),
            None
        )

        # ---------------------------
        # Fallback for Active Market Metrics
        # ---------------------------
        active_one_month = active_df[
            (active_df['postal_code'] == postal_code) &
            (active_df['date_of_asking_price'] >= one_month_ago)
            ]
        if len(active_one_month) > 0:
            current_homes_in_market = len(active_one_month)
        else:
            active_three_months = active_df[
                (active_df['postal_code'] == postal_code) &
                (active_df['date_of_asking_price'] >= three_months_ago)
                ]
            if len(active_three_months) > 0:
                current_homes_in_market = len(active_three_months)
            else:
                active_six_months = active_df[
                    (active_df['postal_code'] == postal_code) &
                    (active_df['date_of_asking_price'] >= six_months_ago)
                    ]
                current_homes_in_market = len(active_six_months)

        active_three_months = active_df[
            (active_df['postal_code'] == postal_code) &
            (active_df['date_of_asking_price'] >= three_months_ago)
            ]
        if len(active_three_months) > 0:
            current_homes_in_market_trend = len(active_three_months)
        else:
            active_six_months = active_df[
                (active_df['postal_code'] == postal_code) &
                (active_df['date_of_asking_price'] >= six_months_ago)
                ]
            current_homes_in_market_trend = len(active_six_months)

        # ---------------------------
        # Historical Sold Metrics from the "records" Collection
        # ---------------------------
        mongodb_data = df.copy()
        mongodb_data['date_of_asking_price'] = pd.to_datetime(
            mongodb_data['date_of_asking_price'],
            dayfirst=True,
            errors='coerce'
        )
        mongodb_data['date_of_sold_price'] = pd.to_datetime(
            mongodb_data['date_of_sold_price'],
            dayfirst=True,
            errors='coerce'
        )
        mongodb_data['postal_code'] = pd.to_numeric(mongodb_data['postal_code'], errors='coerce').fillna(0).astype(int)

        # Define time filters for each period
        one_month_filter = mongodb_data['date_of_sold_price'] >= one_month_ago
        three_month_filter = mongodb_data['date_of_sold_price'] >= three_months_ago
        six_month_filter = mongodb_data['date_of_sold_price'] >= six_months_ago

        # Try hierarchical matching with 1-month data first
        median_price_base, homes_sold = self.market_analyzer.calculate_hierarchical_median_price(
            mongodb_data, input_data, postal_code, sublocality, locality, one_month_filter
        )

        # If not enough data in 1 month, try with 3-month data
        if median_price_base == 0:
            median_price_base, homes_sold = self.market_analyzer.calculate_hierarchical_median_price(
                mongodb_data, input_data, postal_code, sublocality, locality, three_month_filter
            )

        # If still not enough data, try with 6-month data
        if median_price_base == 0:
            median_price_base, homes_sold = self.market_analyzer.calculate_hierarchical_median_price(
                mongodb_data, input_data, postal_code, sublocality, locality, six_month_filter
            )

        # Calculate trend data (previous 3 months vs current 3 months)
        current_3mo = three_month_filter & ~one_month_filter
        previous_3mo = six_month_filter & ~three_month_filter

        median_price_trend, _ = self.market_analyzer.calculate_hierarchical_median_price(
            mongodb_data, input_data, postal_code, sublocality, locality, previous_3mo
        )

        # If trend couldn't be calculated, use the base price
        if median_price_trend == 0:
            median_price_trend = median_price_base

        # Keep the existing code for days on market calculation
        filtered_sold_1_month = mongodb_data[one_month_filter].dropna(
            subset=['date_of_sold_price', 'date_of_asking_price']
        ).copy()

        filtered_sold_3_months = mongodb_data[three_month_filter].dropna(
            subset=['date_of_sold_price', 'date_of_asking_price']
        ).copy()

        filtered_sold_6_months = mongodb_data[six_month_filter].dropna(
            subset=['date_of_sold_price', 'date_of_asking_price']
        ).copy()

        # Calculate days on market
        if not filtered_sold_1_month.empty:
            filtered_sold_1_month["days_on_market"] = (
                    filtered_sold_1_month["date_of_sold_price"] - filtered_sold_1_month["date_of_asking_price"]
            ).dt.days

        if not filtered_sold_3_months.empty:
            filtered_sold_3_months["days_on_market"] = (
                    filtered_sold_3_months["date_of_sold_price"] - filtered_sold_3_months["date_of_asking_price"]
            ).dt.days

        if not filtered_sold_6_months.empty:
            filtered_sold_6_months["days_on_market"] = (
                    filtered_sold_6_months["date_of_sold_price"] - filtered_sold_6_months["date_of_asking_price"]
            ).dt.days

        # Calculate average days on market
        if not filtered_sold_1_month.empty:
            avg_days_base = self.market_analyzer.calc_avg_days_on_market(
                filtered_sold_1_month["days_on_market"].sum(), filtered_sold_1_month.shape[0])
            if not filtered_sold_3_months.empty:
                avg_days_trend = self.market_analyzer.calc_avg_days_on_market(
                    filtered_sold_3_months["days_on_market"].sum(), filtered_sold_3_months.shape[0])
            elif not filtered_sold_6_months.empty:
                avg_days_trend = self.market_analyzer.calc_avg_days_on_market(
                    filtered_sold_6_months["days_on_market"].sum(), filtered_sold_6_months.shape[0])
            else:
                avg_days_trend = avg_days_base
        elif not filtered_sold_3_months.empty:
            avg_days_base = self.market_analyzer.calc_avg_days_on_market(
                filtered_sold_3_months["days_on_market"].sum(), filtered_sold_3_months.shape[0])
            if not filtered_sold_6_months.empty:
                avg_days_trend = self.market_analyzer.calc_avg_days_on_market(
                    filtered_sold_6_months["days_on_market"].sum(), filtered_sold_6_months.shape[0])
            else:
                avg_days_trend = avg_days_base
        elif not filtered_sold_6_months.empty:
            avg_days_base = self.market_analyzer.calc_avg_days_on_market(
                filtered_sold_6_months["days_on_market"].sum(), filtered_sold_6_months.shape[0])
            avg_days_trend = avg_days_base
        else:
            avg_days_base = 0
            avg_days_trend = 0

        # Calculate recommended and estimated prices
        filtered_df_price_calculation = mongodb_data[
            (mongodb_data['postal_code'] == postal_code) &
            (mongodb_data['area'] == sublocality)
            ]

        recommended_price = self.market_analyzer.calculate_recommended_selling_price(
            filtered_df_price_calculation, predicted_result, input_data, active_df
        )

        estimated_price = self.market_analyzer.calculate_estimated_sold_price(
            filtered_df_price_calculation, predicted_result
        )

        # Apply business logic for estimated price
        if estimated_price > recommended_price:
            estimated_price = recommended_price
        if recommended_price - estimated_price > 0.01 * recommended_price:
            estimated_price = int(recommended_price * 0.99)

        # Get property type for comparable properties
        property_type = PropertyTypeClassifier.map_property_type(input_data)

        # Get area for property matching
        area = next(
            (c.get("longText") for c in input_address_components if "area" in c.get("types", [])),
            sublocality
        )

        # Build and return the final response JSON
        return {
            "value_estimate": PricePredictor.round_cents_to_thousand(int(float(predicted_result) * 100)),
            "recommended_selling_price": PricePredictor.round_cents_to_thousand(recommended_price),
            "estimated_sold_price": PricePredictor.round_cents_to_thousand(estimated_price),
            "market_trend": {
                "area_location": input_address_components,
                "median_sale_price": int(median_price_base * 100),
                "median_sale_price_trend": self.market_analyzer.calc_median_trend(median_price_base,
                                                                                  median_price_trend),
                "avg_days_on_market": int(avg_days_base),
                "avg_days_on_market_trend": self.market_analyzer.predict_trend(avg_days_base, avg_days_trend,
                                                                               "avg_days_on_market"),
                "homes_sold": homes_sold,
                "homes_sold_trend": self.market_analyzer.predict_trend(
                    homes_sold,
                    (filtered_sold_3_months.shape[0] if not filtered_sold_3_months.empty else homes_sold)
                ),
                "homes_currently_in_market": current_homes_in_market,
                "homes_currently_in_market_trend": self.market_analyzer.predict_trend(
                    current_homes_in_market,
                    current_homes_in_market_trend
                )
            },
            "avg_days_on_market_over_time": self.market_analyzer.calc_avg_days_on_market_over_time(
                mongodb_data,
                input_data=input_data,
                sublocality=sublocality,
                locality=locality
            ),
            "last_5_sold_properties": self.property_comparator.get_last_5_sold_properties(
                input_data,
                area,
                postal_code,
                property_type
            ),
        }
"""
Property type mappings and classification utilities.
"""
from typing import Dict, Optional
from app.exceptions import InvalidPropertyTypeError


class PropertyTypeMappings:
    """Constants and mappings for property type classification."""

    TYPE_MAP = {
        "apartment": 1,
        "1-storey terraced house": 2,
        "1.5-storey terraced house": 3,
        "2-storey terraced house": 4,
        "2.5-storey terraced house": 5,
        "3-storey terraced house": 6,
        "3.5-storey terraced house": 7,
        "4-storey terraced house": 8,
        "4.5-storey terraced house": 9,
        "bungalow house": 10,
        "link bungalow": 11,
        "zero-lot bungalow": 12,
        "twin villa": 13,
        "twin courtyard villa": 14,
        "cluster house": 15,
        "semi-detached house": 16,
        "townhouse": 17,
        "condominium": 18,
        "condominium penthouse": 19,
        "condominium, penthouse": 19,
        "penthouse": 19,
        "duplex": 20,
        "flat": 21,
        "service residence": 22,
        "studio": 23,
        "townhouse condo": 24,
    }

    CATEGORY_MAP = {
        "apartment": 1,
        "1-storey terraced house": 2,
        "1.5-storey terraced house": 2,
        "2-storey terraced house": 2,
        "2.5-storey terraced house": 2,
        "3-storey terraced house": 2,
        "3.5-storey terraced house": 2,
        "4-storey terraced house": 2,
        "4.5-storey terraced house": 2,
        "bungalow house": 3,
        "link bungalow": 3,
        "zero-lot bungalow": 3,
        "twin villa": 3,
        "twin courtyard villa": 3,
        "cluster house": 4,
        "semi-detached house": 5,
        "townhouse": 6,
        "condominium": 7,
        "condominium penthouse": 7,
        "condominium, penthouse": 7,
        "penthouse": 7,
        "townhouse condo": 7,
        "service residence": 8,
        "duplex": 9,
        "flat": 10,
        "studio": 11,
    }

    AREA_MAP = {
        "desa parkcity": 1,
        "mont kiara": 2,
        "petaling jaya": 3,
        "subang jaya": 4,
        "taman tun dr ismail": 5
    }


class PropertyTypeClassifier:
    """Handles property type classification and mapping."""

    @staticmethod
    def get_property_string_from_code(input_data: Dict) -> Optional[str]:
        """
        Get the property type string based on the property type codes.

        Args:
            input_data: Dictionary containing property type codes

        Returns:
            Property type string (e.g., "condominium", "2-storey terraced house")
        """
        pt1 = input_data.get("propertyType$1")
        if pt1 is None:
            return None

        if pt1 == "0":  # Landed properties
            pt2 = input_data.get("propertyType$2")
            if pt2 is None:
                return None

            if pt2 == "0":  # Bungalow
                pt3 = input_data.get("propertyType$3")
                mapping = {
                    "0": "bungalow house",
                    "1": "link bungalow",
                    "2": "zero-lot bungalow",
                    "3": "twin villa",
                    "4": "twin courtyard villa"
                }
                return mapping.get(pt3)
            elif pt2 == "1":  # Semi-detached
                pt3 = input_data.get("propertyType$3")
                mapping = {"0": "semi-detached house", "1": "cluster house"}
                return mapping.get(pt3)
            elif pt2 == "2":  # Terrace/Link
                pt3 = input_data.get("propertyType$3")
                mapping = {
                    "0": "townhouse",
                    "1": "1-storey terraced house",
                    "2": "1.5-storey terraced house",
                    "3": "2-storey terraced house",
                    "4": "2.5-storey terraced house",
                    "5": "3-storey terraced house",
                    "6": "3.5-storey terraced house",
                    "7": "4-storey terraced house",
                    "8": "4.5-storey terraced house",
                }
                return mapping.get(pt3)
        elif pt1 == "1":  # High-rise properties
            pt2 = input_data.get("propertyType$2")
            if pt2 is None:
                return None

            mapping = {
                "0": "condominium",
                "1": "apartment",
                "2": "flat",
                "3": "penthouse",
                "4": "service residence",
                "5": "studio",
                "6": "duplex",
                "7": "townhouse condo"
            }
            return mapping.get(pt2)

        return None

    @staticmethod
    def map_property_type(input_data: Dict) -> int:
        """
        Map property type codes to numeric property type values.

        Args:
            input_data: Dictionary containing property type codes

        Returns:
            int: Numeric property type code

        Raises:
            InvalidPropertyTypeError: If property type cannot be determined
        """
        property_str = PropertyTypeClassifier.get_property_string_from_code(input_data)

        if property_str is None:
            raise InvalidPropertyTypeError("Could not determine property type from input.")

        mapped_value = PropertyTypeMappings.TYPE_MAP.get(property_str.lower())
        if mapped_value is None:
            raise InvalidPropertyTypeError(f"Property type '{property_str}' not found in mapping.")

        return mapped_value

    @staticmethod
    def get_property_category(input_data: Dict) -> Optional[int]:
        """
        Determine the property category based on the property type.

        Args:
            input_data: The input data dictionary

        Returns:
            int: Property category code (1-11) or None if not found
        """
        property_str = PropertyTypeClassifier.get_property_string_from_code(input_data)
        if property_str is None:
            return None

        return PropertyTypeMappings.CATEGORY_MAP.get(property_str.lower())

    @staticmethod
    def is_property_landed(input_data: Dict) -> int:
        """
        Determine if a property is landed based on propertyType$1.

        Args:
            input_data: Property data dictionary

        Returns:
            int: 1 if landed, 0 if not
        """
        property_type = input_data.get("propertyType$1")
        # "0" is for landed properties
        return 1 if property_type == "0" else 0


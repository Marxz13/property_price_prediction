"""
Configuration settings and constants for property valuation.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient


class Config:
    """Central configuration class for property valuation system."""

    def __init__(self):
        """Initialize configuration with environment variables."""
        load_dotenv()
        self.mongodb_uri = os.environ.get("MONGODB_URI")
        self.numeric_precision = 3

    def get_mongodb_client(self) -> MongoClient:
        """
        Get MongoDB client using configured URI.

        Returns:
            MongoClient: Configured MongoDB client
        """
        return MongoClient(self.mongodb_uri)

    def get_database(self, db_name: str) -> Any:
        """
        Get specific MongoDB database.

        Args:
            db_name: Name of the database

        Returns:
            Database: MongoDB database instance
        """
        client = self.get_mongodb_client()
        return client[db_name]

    def get_collection(self, db_name: str, collection_name: str) -> Any:
        """
        Get specific MongoDB collection.

        Args:
            db_name: Name of the database
            collection_name: Name of the collection

        Returns:
            Collection: MongoDB collection instance
        """
        db = self.get_database(db_name)
        return db[collection_name]


# Create global config instance
config = Config()
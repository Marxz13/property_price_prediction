"""
WSGI entry point for property valuation Flask application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from app import create_app

# Create the Flask application instance
app = create_app()

if __name__ == "__main__":
    # Run the app in debug mode if running directly
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"

    app.run(host=host, port=port, debug=debug)
"""
Flask application factory for property valuation system.
"""
import os
from flask import Flask
from flask_cors import CORS


def create_app(test_config=None):
    """
    Create and configure the Flask application.

    Args:
        test_config: Test configuration to override default configs

    Returns:
        Flask application instance
    """
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # Enable CORS
    CORS(app)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev"),
        MONGODB_URI=os.environ.get("MONGODB_URI", "mongodb://localhost:27017/"),
        MODEL_DIR=os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")),
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Register blueprints
    from app.routes import valuation, market_analysis

    app.register_blueprint(valuation.bp)
    app.register_blueprint(market_analysis.bp)

    # Create a simple route to check if the app is running
    @app.route("/health")
    def health_check():
        return {"status": "ok"}

    return app
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
    app = Flask(__name__, instance_relative_config=True)

    CORS(app)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev"),
        MONGODB_URI=os.environ.get("MONGODB_URI", "mongodb://localhost:27017/"),
        MODEL_DIR=os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")),
    )

    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from app.routes import valuation

    app.register_blueprint(valuation.bp)

    @app.route("/health")
    def health_check():
        return {"status": "ok"}

    return app
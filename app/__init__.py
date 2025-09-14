from flask import Flask
from .routes.home import home_bp
from .routes.predict_image import image_bp
from .routes.predict_fertilizer import fert_bp

def create_app():
    app = Flask(__name__)

    # Blueprints
    app.register_blueprint(home_bp)
    app.register_blueprint(image_bp, url_prefix="/")
    app.register_blueprint(fert_bp, url_prefix="/")

    return app

"""
AI Interior Design Studio - Application Factory

This module contains the Flask application factory and configuration.
"""

import os
import logging
from flask import Flask, jsonify
from dotenv import load_dotenv

from .config import Config

# Load environment variables
load_dotenv()


def create_app(config_class=Config):
    """
    Application factory for creating Flask app instances.
    
    Args:
        config_class: Configuration class to use
        
    Returns:
        Flask app instance
    """
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)
    
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['WEIGHTS_FOLDER'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    setup_logging(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'gpu_available': Config.USE_GPU
        })
    
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Welcome to AI Interior Design Studio API',
            'version': '1.0.0',
            'documentation': '/docs',
            'endpoints': {
                'wall_color': '/api/wall-color',
                'style_transfer': '/api/style-transfer',
                'object_detection': '/api/objects',
                'inpainting': '/api/inpaint',
                'recommendations': '/api/recommendations',
                'wallpaper': '/api/wallpaper',
                'tiles': '/api/tiles'
            }
        })
    
    return app


def setup_logging(app):
    """Configure application logging."""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log', encoding='utf-8')
        ]
    )
    app.logger.setLevel(getattr(logging, log_level))


def register_blueprints(app):
    """Register all API blueprints."""
    from .api.wall_color import wall_color_bp
    from .api.style_transfer import style_transfer_bp
    from .api.object_detection import object_detection_bp
    from .api.inpainting import inpainting_bp
    from .api.recommendations import recommendations_bp
    from .api.wallpaper import wallpaper_bp
    from .api.tiles import tiles_bp
    
    app.register_blueprint(wall_color_bp, url_prefix='/api/wall-color')
    app.register_blueprint(style_transfer_bp, url_prefix='/api/style-transfer')
    app.register_blueprint(object_detection_bp, url_prefix='/api/objects')
    app.register_blueprint(inpainting_bp, url_prefix='/api/inpaint')
    app.register_blueprint(recommendations_bp, url_prefix='/api/recommendations')
    app.register_blueprint(wallpaper_bp, url_prefix='/api/wallpaper')
    app.register_blueprint(tiles_bp, url_prefix='/api/tiles')


def register_error_handlers(app):
    """Register error handlers."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': str(error.description)
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found'
        }), 404
    
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({
            'error': 'File Too Large',
            'message': f'File size exceeds maximum limit of {Config.MAX_UPLOAD_SIZE_MB}MB'
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Internal server error: {error}')
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500

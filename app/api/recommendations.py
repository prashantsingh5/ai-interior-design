"""
Image Recommendations API Endpoints.

This module provides endpoints for intelligent interior design
recommendations based on visual similarity using ResNet50.
"""

import os
import logging
import numpy as np
from flask import Blueprint, request, jsonify

from ..config import Config
from ..services.recommendation import RecommendationService

logger = logging.getLogger(__name__)
recommendations_bp = Blueprint('recommendations', __name__)

# Initialize recommendation service
recommendation_service = None


def get_recommendation_service():
    """Get or initialize the recommendation service."""
    global recommendation_service
    if recommendation_service is None:
        recommendation_service = RecommendationService()
    return recommendation_service


@recommendations_bp.route('/similar', methods=['POST'])
def get_similar_images():
    """
    Get similar interior design images.
    
    Request (multipart/form-data):
        - image: Reference room image
        - category: (optional) Room type (Bedroom, Kitchen, Bathroom, Livingroom, Dinning)
        - limit: (optional) Number of recommendations (default: 5)
        
    Returns:
        JSON list of recommended images with similarity scores
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not Config.is_allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        category = request.form.get('category', 'Bedroom')
        limit = int(request.form.get('limit', 5))
        
        # Get recommendations
        service = get_recommendation_service()
        results = service.get_recommendations(
            image_file=file,
            category=category,
            limit=limit
        )
        
        logger.info(f"Generated {len(results['recommendations'])} recommendations for category: {category}")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': f'Recommendation failed: {str(e)}'}), 500


@recommendations_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get available room categories."""
    categories = [
        {'name': 'Bedroom', 'description': 'Bedroom interior designs'},
        {'name': 'Kitchen', 'description': 'Kitchen interior designs'},
        {'name': 'Bathroom', 'description': 'Bathroom interior designs'},
        {'name': 'Livingroom', 'description': 'Living room interior designs'},
        {'name': 'Dinning', 'description': 'Dining room interior designs'}
    ]
    return jsonify({'categories': categories})


@recommendations_bp.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze an interior image for its visual characteristics.
    
    Request (multipart/form-data):
        - image: Room image file
        
    Returns:
        JSON with dominant colors, contrast, and style features
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        service = get_recommendation_service()
        analysis = service.analyze_image(file)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

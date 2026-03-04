"""
Wall Color Change API Endpoints.

This module provides endpoints for AI-powered wall color visualization
using SAM (Segment Anything Model) and GroundingDINO.
"""

import os
import logging
from flask import Blueprint, request, send_file, jsonify, current_app
from werkzeug.utils import secure_filename

from ..config import Config
from ..utils.image_processing import save_uploaded_file, get_output_path

logger = logging.getLogger(__name__)
wall_color_bp = Blueprint('wall_color', __name__)

# Lazy import for SegmentationService
_segmentation_service = None

def get_segmentation_service():
    """Get or create SegmentationService instance."""
    global _segmentation_service
    if _segmentation_service is None:
        from ..services.segmentation import SegmentationService
        _segmentation_service = SegmentationService()
    return _segmentation_service


@wall_color_bp.route('/change', methods=['POST'])
def change_wall_color():
    """
    Change wall color in an interior image.
    
    Request (multipart/form-data):
        - image: Room image file
        - text_prompt: Wall description (e.g., "wall", "back wall")
        - color_name: Target color name (e.g., "light blue", "beige")
        
    Returns:
        Image file with changed wall color
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not Config.is_allowed_file(image.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        text_prompt = request.form.get('text_prompt', 'wall')
        color_name = request.form.get('color_name', 'white')
        
        # Save uploaded file
        input_path = save_uploaded_file(image, 'wall_color')
        output_path = get_output_path('wall_color', 'result.png')
        
        # Process image
        service = get_segmentation_service()
        result_path = service.change_wall_color(
            image_path=input_path,
            text_prompt=text_prompt,
            color_name=color_name,
            output_path=output_path
        )
        
        logger.info(f"Wall color changed successfully: {result_path}")
        
        return send_file(
            result_path,
            mimetype='image/png',
            as_attachment=True,
            download_name='wall_color_result.png'
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error changing wall color: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@wall_color_bp.route('/colors', methods=['GET'])
def get_available_colors():
    """
    Get list of available color names.
    
    Returns:
        JSON list of available color names and their RGB values
    """
    colors = [
        {'name': name, 'rgb': list(rgb)}
        for name, rgb in Config.COLOR_MAP.items()
    ]
    return jsonify({
        'colors': colors,
        'total': len(colors)
    })

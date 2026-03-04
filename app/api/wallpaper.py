"""
Wallpaper Application API Endpoints.

This module provides endpoints for virtually applying
wallpaper patterns to walls using AI segmentation.
"""

import os
import logging
from flask import Blueprint, request, send_file, jsonify

from ..config import Config
from ..utils.image_processing import save_uploaded_file, get_output_path

logger = logging.getLogger(__name__)
wallpaper_bp = Blueprint('wallpaper', __name__)

# Lazy import for SegmentationService
_segmentation_service = None

def get_segmentation_service():
    """Get or create SegmentationService instance."""
    global _segmentation_service
    if _segmentation_service is None:
        from ..services.segmentation import SegmentationService
        _segmentation_service = SegmentationService()
    return _segmentation_service


@wallpaper_bp.route('/apply', methods=['POST'])
def apply_wallpaper():
    """
    Apply wallpaper pattern to walls in an image.
    
    Request (multipart/form-data):
        - room_image: Room image file
        - wallpaper_image: Wallpaper pattern image
        
    Returns:
        Image file with wallpaper applied
    """
    try:
        # Validate request
        if 'room_image' not in request.files:
            return jsonify({'error': 'No room image provided'}), 400
        
        if 'wallpaper_image' not in request.files:
            return jsonify({'error': 'No wallpaper image provided'}), 400
        
        room_image = request.files['room_image']
        wallpaper_image = request.files['wallpaper_image']
        
        if room_image.filename == '' or wallpaper_image.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded files
        room_path = save_uploaded_file(room_image, 'wallpaper', prefix='room_')
        wallpaper_path = save_uploaded_file(wallpaper_image, 'wallpaper', prefix='pattern_')
        output_path = get_output_path('wallpaper', 'result.jpg')
        
        # Process
        service = get_segmentation_service()
        result_path = service.apply_wallpaper(
            room_image_path=room_path,
            wallpaper_image_path=wallpaper_path,
            output_path=output_path
        )
        
        logger.info("Wallpaper applied successfully")
        
        return send_file(
            result_path,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='wallpaper_result.jpg'
        )
        
    except Exception as e:
        logger.error(f"Wallpaper application error: {e}")
        return jsonify({'error': f'Wallpaper application failed: {str(e)}'}), 500


@wallpaper_bp.route('/info', methods=['GET'])
def get_info():
    """Get information about the wallpaper service."""
    return jsonify({
        'service': 'Wallpaper Application',
        'description': 'Apply wallpaper patterns to walls using AI segmentation',
        'supported_formats': list(Config.ALLOWED_EXTENSIONS),
        'tips': [
            'Use high-resolution images for best results',
            'Wallpaper patterns work best with repeating textures',
            'Ensure walls are clearly visible in the room image'
        ]
    })

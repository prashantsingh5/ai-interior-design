"""
Object Detection API Endpoints.

This module provides endpoints for detecting furniture and
interior objects using GroundingDINO.
"""

import logging
from flask import Blueprint, request, jsonify

from ..config import Config
from ..utils.image_processing import save_uploaded_file

logger = logging.getLogger(__name__)
object_detection_bp = Blueprint('object_detection', __name__)

# Lazy import for SegmentationService
_segmentation_service = None

def get_segmentation_service():
    """Get or create SegmentationService instance."""
    global _segmentation_service
    if _segmentation_service is None:
        from ..services.segmentation import SegmentationService
        _segmentation_service = SegmentationService()
    return _segmentation_service


@object_detection_bp.route('/detect', methods=['POST'])
def detect_objects():
    """
    Detect objects in an interior image.
    
    Request (multipart/form-data):
        - file: Room image file
        - custom_objects: (optional) Comma-separated list of objects to detect
        
    Returns:
        JSON list of detected objects with confidence scores
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not Config.is_allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get optional custom objects
        custom_objects = request.form.get('custom_objects')
        objects_to_detect = None
        if custom_objects:
            objects_to_detect = [obj.strip() for obj in custom_objects.split(',')]
        
        # Save uploaded file
        input_path = save_uploaded_file(file, 'detection')
        
        # Detect objects
        service = get_segmentation_service()
        detected_objects = service.detect_objects(
            image_path=input_path,
            objects_to_detect=objects_to_detect
        )
        
        logger.info(f"Detected {len(detected_objects)} objects")
        
        return jsonify({
            'detected_objects': detected_objects,
            'total': len(detected_objects)
        })
        
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@object_detection_bp.route('/default-objects', methods=['GET'])
def get_default_objects():
    """Get list of default detectable objects."""
    return jsonify({
        'objects': Config.DEFAULT_DETECTABLE_OBJECTS,
        'total': len(Config.DEFAULT_DETECTABLE_OBJECTS)
    })

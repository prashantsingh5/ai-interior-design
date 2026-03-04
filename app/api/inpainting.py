"""
AI Inpainting API Endpoints.

This module provides endpoints for AI-powered object replacement
using Stable Diffusion Inpainting.
"""

import logging
from flask import Blueprint, request, send_file, jsonify

from ..config import Config
from ..utils.image_processing import save_uploaded_file, get_output_path

logger = logging.getLogger(__name__)
inpainting_bp = Blueprint('inpainting', __name__)

# Lazy import for InpaintingService
_inpainting_service = None

def get_inpainting_service():
    """Get or create InpaintingService instance."""
    global _inpainting_service
    if _inpainting_service is None:
        from ..services.inpainting import InpaintingService
        _inpainting_service = InpaintingService()
    return _inpainting_service


@inpainting_bp.route('/apply', methods=['POST'])
def inpaint_object():
    """
    Replace an object in the image using AI inpainting.
    
    Request (multipart/form-data):
        - image: Room image file
        - object_to_replace: Object to detect and replace (e.g., "sofa", "chair")
        - replacement_prompt: Description of the replacement (e.g., "modern blue sofa")
        
    Returns:
        Inpainted image file
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
        
        object_to_replace = request.form.get('object_to_replace')
        replacement_prompt = request.form.get('replacement_prompt')
        
        if not object_to_replace or not replacement_prompt:
            return jsonify({
                'error': 'Missing required parameters: object_to_replace and replacement_prompt'
            }), 400
        
        # Save uploaded file
        input_path = save_uploaded_file(image, 'inpainting')
        output_path = get_output_path('inpainting', 'inpainted.png')
        
        # Process image
        service = get_inpainting_service()
        result_path = service.inpaint(
            image_path=input_path,
            object_to_detect=object_to_replace,
            inpaint_prompt=replacement_prompt,
            output_path=output_path
        )
        
        logger.info(f"Inpainting completed: replaced '{object_to_replace}'")
        
        return send_file(
            result_path,
            mimetype='image/png',
            as_attachment=True,
            download_name='inpainted_result.png'
        )
        
    except Exception as e:
        logger.error(f"Inpainting error: {e}")
        return jsonify({'error': f'Inpainting failed: {str(e)}'}), 500


@inpainting_bp.route('/info', methods=['GET'])
def get_info():
    """Get information about the inpainting service."""
    return jsonify({
        'service': 'AI Inpainting',
        'model': 'Stable Diffusion 2.0 Inpainting',
        'device': Config.DEVICE,
        'example_prompts': [
            'modern minimalist sofa',
            'wooden dining table',
            'elegant floor lamp',
            'cozy armchair with throw blanket',
            'contemporary coffee table'
        ]
    })

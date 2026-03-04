"""
Neural Style Transfer API Endpoints.

This module provides endpoints for applying artistic styles
to interior images using VGG19-based neural style transfer.
"""

import logging
from io import BytesIO
from flask import Blueprint, request, send_file, jsonify

from ..config import Config

logger = logging.getLogger(__name__)
style_transfer_bp = Blueprint('style_transfer', __name__)

# Lazy import for StyleTransferService
_style_transfer_service = None

def get_style_transfer_service():
    """Get or create StyleTransferService instance."""
    global _style_transfer_service
    if _style_transfer_service is None:
        from ..services.style_transfer import StyleTransferService
        _style_transfer_service = StyleTransferService()
    return _style_transfer_service


@style_transfer_bp.route('/apply', methods=['POST'])
def apply_style_transfer():
    """
    Apply neural style transfer to an image.
    
    Request (multipart/form-data):
        - content_image: The interior/room image
        - style_image: The style reference image
        - epochs: (optional) Number of optimization epochs (default: 60)
        - style_weight: (optional) Style weight (default: 1e6)
        
    Returns:
        Stylized image file
    """
    try:
        # Validate request
        if 'content_image' not in request.files:
            return jsonify({'error': 'No content image provided'}), 400
        
        if 'style_image' not in request.files:
            return jsonify({'error': 'No style image provided'}), 400
        
        content_file = request.files['content_image']
        style_file = request.files['style_image']
        
        if content_file.filename == '' or style_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get optional parameters
        epochs = int(request.form.get('epochs', 60))
        style_weight = float(request.form.get('style_weight', 1e6))
        
        # Process images
        service = get_style_transfer_service()
        output_image = service.transfer_style(
            content_image=content_file.read(),
            style_image=style_file.read(),
            epochs=epochs,
            style_weight=style_weight
        )
        
        # Convert to bytes
        output_buffer = BytesIO()
        output_image.save(output_buffer, format="JPEG", quality=95)
        output_buffer.seek(0)
        
        logger.info("Style transfer completed successfully")
        
        return send_file(
            output_buffer,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='styled_interior.jpg'
        )
        
    except Exception as e:
        logger.error(f"Style transfer error: {e}")
        return jsonify({'error': f'Style transfer failed: {str(e)}'}), 500


@style_transfer_bp.route('/info', methods=['GET'])
def get_info():
    """Get information about the style transfer service."""
    return jsonify({
        'service': 'Neural Style Transfer',
        'model': 'VGG19',
        'default_epochs': 60,
        'device': Config.DEVICE,
        'supported_formats': list(Config.ALLOWED_EXTENSIONS)
    })

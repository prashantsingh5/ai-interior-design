"""
Tile Gallery API Endpoints.

This module provides endpoints for browsing and retrieving
tile images for flooring and wall applications.
"""

import os
import io
import base64
import logging
from flask import Blueprint, jsonify, send_file

from ..config import Config

logger = logging.getLogger(__name__)
tiles_bp = Blueprint('tiles', __name__)

# Tile configuration
TILE_SIZE = (150, 150)
TILES_FOLDER = os.path.join(Config.DATA_FOLDER, 'tiles')


def get_tiles_folder():
    """Get the tiles folder path, creating it if necessary."""
    os.makedirs(TILES_FOLDER, exist_ok=True)
    return TILES_FOLDER


@tiles_bp.route('/gallery', methods=['GET'])
def get_tile_gallery():
    """
    Get all available tile images as base64 encoded data.
    
    Returns:
        JSON list of tile images with metadata
    """
    try:
        from PIL import Image
        
        tiles_folder = get_tiles_folder()
        tiles_data = []
        
        if not os.path.exists(tiles_folder):
            return jsonify({
                'success': True,
                'images': [],
                'total': 0,
                'message': 'No tiles available. Add tile images to the data/tiles folder.'
            })
        
        valid_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        
        for idx, filename in enumerate(sorted(os.listdir(tiles_folder))):
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            
            if ext in valid_extensions:
                filepath = os.path.join(tiles_folder, filename)
                
                try:
                    with Image.open(filepath) as img:
                        # Resize for thumbnail
                        img_resized = img.resize(TILE_SIZE, Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        img_resized.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        tiles_data.append({
                            'index': idx + 1,
                            'name': filename,
                            'image': f"data:image/png;base64,{img_base64}",
                            'dimensions': list(TILE_SIZE)
                        })
                except Exception as e:
                    logger.warning(f"Could not process tile image {filename}: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'images': tiles_data,
            'total': len(tiles_data),
            'tile_size': list(TILE_SIZE)
        })
        
    except Exception as e:
        logger.error(f"Error fetching tile gallery: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@tiles_bp.route('/image/<filename>', methods=['GET'])
def get_tile_image(filename):
    """
    Get a specific tile image by filename.
    
    Args:
        filename: Name of the tile image file
        
    Returns:
        Image file
    """
    try:
        tiles_folder = get_tiles_folder()
        filepath = os.path.join(tiles_folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Tile not found'}), 404
        
        return send_file(filepath)
        
    except Exception as e:
        logger.error(f"Error fetching tile image: {e}")
        return jsonify({'error': str(e)}), 500


@tiles_bp.route('/info', methods=['GET'])
def get_info():
    """Get information about the tiles service."""
    tiles_folder = get_tiles_folder()
    tile_count = 0
    
    if os.path.exists(tiles_folder):
        valid_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        tile_count = sum(
            1 for f in os.listdir(tiles_folder)
            if '.' in f and f.rsplit('.', 1)[-1].lower() in valid_extensions
        )
    
    return jsonify({
        'service': 'Tile Gallery',
        'available_tiles': tile_count,
        'thumbnail_size': list(TILE_SIZE),
        'supported_formats': ['png', 'jpg', 'jpeg', 'gif', 'webp']
    })

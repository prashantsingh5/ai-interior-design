"""
Image Processing Utilities.

This module provides common image processing functions
used across the application.
"""

import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from ..config import Config


def save_uploaded_file(file, subfolder: str, prefix: str = '') -> str:
    """
    Save uploaded file to media folder.
    
    Args:
        file: Uploaded file object
        subfolder: Subfolder within media folder
        prefix: Optional prefix for filename
        
    Returns:
        Path to saved file
    """
    # Create folder structure
    upload_folder = os.path.join(Config.UPLOAD_FOLDER, subfolder, 'input')
    os.makedirs(upload_folder, exist_ok=True)
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    unique_name = f"{prefix}{name}_{uuid.uuid4().hex[:8]}{ext}"
    
    filepath = os.path.join(upload_folder, unique_name)
    file.save(filepath)
    
    return filepath


def get_output_path(subfolder: str, filename: str) -> str:
    """
    Generate output path for processed files.
    
    Args:
        subfolder: Subfolder within media folder
        filename: Output filename
        
    Returns:
        Path for output file
    """
    output_folder = os.path.join(Config.UPLOAD_FOLDER, subfolder, 'output')
    os.makedirs(output_folder, exist_ok=True)
    
    # Add timestamp to filename
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{name}_{timestamp}{ext}"
    
    return os.path.join(output_folder, unique_filename)


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.
    
    Args:
        filename: Filename to check
        
    Returns:
        True if allowed, False otherwise
    """
    return Config.is_allowed_file(filename)


def load_image(path: str) -> Image.Image:
    """
    Load image from path.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image object
    """
    return Image.open(path).convert('RGB')


def save_image(image, path: str, quality: int = 95):
    """
    Save image to path.
    
    Args:
        image: PIL Image or numpy array
        path: Output path
        quality: JPEG quality (if applicable)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Determine format from extension
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.jpg', '.jpeg']:
        image.save(path, 'JPEG', quality=quality)
    elif ext == '.png':
        image.save(path, 'PNG')
    else:
        image.save(path)


def resize_image(image, max_size: int = 1024, maintain_aspect: bool = True):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        ratio = max_size / max(image.size)
        if ratio < 1:
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    
    return image


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    import base64
    
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        PIL Image object
    """
    import base64
    from io import BytesIO
    
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


def get_image_info(path: str) -> dict:
    """
    Get image metadata.
    
    Args:
        path: Path to image file
        
    Returns:
        Dictionary with image info
    """
    image = Image.open(path)
    
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format,
        'size_bytes': os.path.getsize(path)
    }

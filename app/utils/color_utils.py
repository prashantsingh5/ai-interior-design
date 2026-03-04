"""
Color Utilities.

This module provides color manipulation and conversion functions.
"""

from typing import Tuple, List
import numpy as np

from ..config import Config


def get_color_rgb(color_name: str) -> Tuple[int, int, int]:
    """
    Get RGB value for a color name.
    
    Args:
        color_name: Name of the color
        
    Returns:
        Tuple of (R, G, B) values
    """
    return Config.get_color_rgb(color_name)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB to hex color code.
    
    Args:
        rgb: Tuple of (R, G, B) values
        
    Returns:
        Hex color string (e.g., '#FF5733')
    """
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color code to RGB.
    
    Args:
        hex_color: Hex color string (e.g., '#FF5733' or 'FF5733')
        
    Returns:
        Tuple of (R, G, B) values
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB to HSV color space.
    
    Args:
        rgb: Tuple of (R, G, B) values (0-255)
        
    Returns:
        Tuple of (H, S, V) values (0-360, 0-100, 0-100)
    """
    r, g, b = [x / 255.0 for x in rgb]
    
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    
    # Hue
    if diff == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation
    s = 0 if max_c == 0 else (diff / max_c) * 100
    
    # Value
    v = max_c * 100
    
    return (h, s, v)


def hsv_to_rgb(hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Convert HSV to RGB color space.
    
    Args:
        hsv: Tuple of (H, S, V) values (0-360, 0-100, 0-100)
        
    Returns:
        Tuple of (R, G, B) values (0-255)
    """
    h, s, v = hsv
    s, v = s / 100, v / 100
    
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255)
    )


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB to CIE Lab color space.
    
    Args:
        rgb: Tuple of (R, G, B) values (0-255)
        
    Returns:
        Tuple of (L, a, b) values
    """
    import cv2
    
    # Create a 1x1 image with the color
    color = np.array([[rgb]], dtype=np.uint8)
    
    # Convert to Lab
    lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)
    
    return tuple(lab[0, 0].astype(float))


def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Calculate Euclidean distance between two colors in RGB space.
    
    Args:
        color1: First RGB color
        color2: Second RGB color
        
    Returns:
        Distance value (0-441.67)
    """
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))


def complementary_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Get complementary color.
    
    Args:
        rgb: Tuple of (R, G, B) values
        
    Returns:
        Complementary RGB color
    """
    return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])


def adjust_brightness(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """
    Adjust color brightness.
    
    Args:
        rgb: Tuple of (R, G, B) values
        factor: Brightness factor (>1 brighter, <1 darker)
        
    Returns:
        Adjusted RGB color
    """
    return tuple(
        max(0, min(255, int(c * factor)))
        for c in rgb
    )


def blend_colors(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    ratio: float = 0.5
) -> Tuple[int, int, int]:
    """
    Blend two colors.
    
    Args:
        color1: First RGB color
        color2: Second RGB color
        ratio: Blend ratio (0 = color1, 1 = color2)
        
    Returns:
        Blended RGB color
    """
    return tuple(
        int(c1 * (1 - ratio) + c2 * ratio)
        for c1, c2 in zip(color1, color2)
    )


def generate_color_scheme(
    base_color: Tuple[int, int, int],
    scheme: str = 'complementary'
) -> List[Tuple[int, int, int]]:
    """
    Generate color scheme from base color.
    
    Args:
        base_color: Base RGB color
        scheme: Type of scheme ('complementary', 'analogous', 'triadic', 'split')
        
    Returns:
        List of RGB colors in the scheme
    """
    hsv = rgb_to_hsv(base_color)
    h, s, v = hsv
    
    colors = [base_color]
    
    if scheme == 'complementary':
        colors.append(hsv_to_rgb(((h + 180) % 360, s, v)))
        
    elif scheme == 'analogous':
        colors.append(hsv_to_rgb(((h + 30) % 360, s, v)))
        colors.append(hsv_to_rgb(((h - 30) % 360, s, v)))
        
    elif scheme == 'triadic':
        colors.append(hsv_to_rgb(((h + 120) % 360, s, v)))
        colors.append(hsv_to_rgb(((h + 240) % 360, s, v)))
        
    elif scheme == 'split':
        colors.append(hsv_to_rgb(((h + 150) % 360, s, v)))
        colors.append(hsv_to_rgb(((h + 210) % 360, s, v)))
    
    return colors

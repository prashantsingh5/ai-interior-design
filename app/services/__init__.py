"""
Services for AI Interior Design Studio.

Services are loaded lazily to handle missing dependencies gracefully.
"""

import logging

logger = logging.getLogger(__name__)

# Lazy imports to handle missing dependencies
_segmentation_service = None
_style_transfer_service = None
_inpainting_service = None
_recommendation_service = None


def get_segmentation_service():
    """Get or create SegmentationService instance."""
    global _segmentation_service
    if _segmentation_service is None:
        from .segmentation import SegmentationService
        _segmentation_service = SegmentationService
    return _segmentation_service


def get_style_transfer_service():
    """Get or create StyleTransferService class."""
    global _style_transfer_service
    if _style_transfer_service is None:
        from .style_transfer import StyleTransferService
        _style_transfer_service = StyleTransferService
    return _style_transfer_service


def get_inpainting_service():
    """Get or create InpaintingService class."""
    global _inpainting_service
    if _inpainting_service is None:
        from .inpainting import InpaintingService
        _inpainting_service = InpaintingService
    return _inpainting_service


def get_recommendation_service():
    """Get or create RecommendationService class."""
    global _recommendation_service
    if _recommendation_service is None:
        from .recommendation import RecommendationService
        _recommendation_service = RecommendationService
    return _recommendation_service


# For backward compatibility, try direct imports
try:
    from .segmentation import SegmentationService
except ImportError as e:
    logger.warning(f"SegmentationService not available: {e}")
    SegmentationService = None

try:
    from .style_transfer import StyleTransferService
except ImportError as e:
    logger.warning(f"StyleTransferService not available: {e}")
    StyleTransferService = None

try:
    from .inpainting import InpaintingService
except ImportError as e:
    logger.warning(f"InpaintingService not available: {e}")
    InpaintingService = None

try:
    from .recommendation import RecommendationService
except ImportError as e:
    logger.warning(f"RecommendationService not available: {e}")
    RecommendationService = None


__all__ = [
    'SegmentationService',
    'StyleTransferService',
    'InpaintingService',
    'RecommendationService',
    'get_segmentation_service',
    'get_style_transfer_service',
    'get_inpainting_service',
    'get_recommendation_service'
]

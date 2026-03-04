"""
Segmentation Service.

This module provides wall detection, color change, and wallpaper
application using SAM and GroundingDINO models.
"""

import os
import logging
import numpy as np
import cv2
from PIL import Image
from typing import List, Optional, Tuple

# Handle optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from segment_anything import build_sam, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    import groundingdino
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

from ..config import Config

logger = logging.getLogger(__name__)


class SegmentationService:
    """Service for image segmentation tasks using SAM and GroundingDINO."""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        """Singleton pattern for efficient model loading."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the segmentation service."""
        if not SegmentationService._models_loaded:
            self.device = Config.DEVICE
            self.groundingdino_model = None
            self.sam_predictor = None
            logger.info(f"Segmentation service initialized on device: {self.device}")
    
    def _load_grounding_dino(self):
        """Load GroundingDINO model lazily."""
        if self.groundingdino_model is not None:
            return self.groundingdino_model
        
        try:
            from huggingface_hub import hf_hub_download
            
            # Import GroundingDINO modules
            import sys
            grounding_dino_path = os.path.join(Config.WEIGHTS_FOLDER, 'GroundingDINO')
            if os.path.exists(grounding_dino_path):
                sys.path.insert(0, grounding_dino_path)
            
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
            from groundingdino.models import build_model
            
            # Download config and weights
            cache_config = hf_hub_download(
                repo_id=Config.GROUNDING_DINO_REPO,
                filename=Config.GROUNDING_DINO_SWINB_CONFIG
            )
            cache_weights = hf_hub_download(
                repo_id=Config.GROUNDING_DINO_REPO,
                filename=Config.GROUNDING_DINO_SWINB_CKPT
            )
            
            # Build model
            args = SLConfig.fromfile(cache_config)
            args.device = self.device
            model = build_model(args)
            
            # Load weights
            checkpoint = torch.load(cache_weights, map_location=self.device)
            model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            model.to(self.device)
            model.eval()
            
            self.groundingdino_model = model
            logger.info("GroundingDINO model loaded successfully")
            return model
            
        except ImportError as e:
            logger.error(f"GroundingDINO not available: {e}")
            raise ImportError(
                "GroundingDINO is not installed. Please run: pip install groundingdino"
            )
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO: {e}")
            raise
    
    def _load_sam(self):
        """Load SAM model lazily."""
        if self.sam_predictor is not None:
            return self.sam_predictor
        
        try:
            from segment_anything import build_sam, SamPredictor
            
            # Find SAM checkpoint
            sam_checkpoint = self._find_sam_checkpoint()
            
            # Build SAM
            sam_model = build_sam(checkpoint=sam_checkpoint)
            sam_model.to(self.device)
            
            self.sam_predictor = SamPredictor(sam_model)
            logger.info("SAM model loaded successfully")
            return self.sam_predictor
            
        except ImportError as e:
            logger.error(f"SAM not available: {e}")
            raise ImportError(
                "segment-anything is not installed. Please run: pip install segment-anything"
            )
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            raise
    
    def _find_sam_checkpoint(self) -> str:
        """Find SAM checkpoint in weights folder."""
        checkpoint_name = Config.SAM_CHECKPOINT
        
        # Check weights folder
        checkpoint_path = os.path.join(Config.WEIGHTS_FOLDER, checkpoint_name)
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        
        # Search recursively
        for root, dirs, files in os.walk(Config.BASE_DIR):
            if checkpoint_name in files:
                return os.path.join(root, checkpoint_name)
        
        raise FileNotFoundError(
            f"SAM checkpoint '{checkpoint_name}' not found. "
            f"Please download it and place in {Config.WEIGHTS_FOLDER}"
        )
    
    def change_wall_color(
        self,
        image_path: str,
        text_prompt: str,
        color_name: str,
        output_path: str,
        box_threshold: float = None,
        text_threshold: float = None
    ) -> str:
        """
        Change wall color in an image.
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of walls to detect
            color_name: Target color name
            output_path: Path to save output image
            box_threshold: Detection box threshold
            text_threshold: Detection text threshold
            
        Returns:
            Path to output image
        """
        box_threshold = box_threshold or Config.BOX_THRESHOLD
        text_threshold = text_threshold or Config.TEXT_THRESHOLD
        
        # Load models
        groundingdino = self._load_grounding_dino()
        sam_predictor = self._load_sam()
        
        from groundingdino.util import box_ops
        from groundingdino.util.inference import load_image, predict
        
        # Load image
        image_source, image_tensor = load_image(image_path)
        
        # Detect walls
        boxes, logits, phrases = predict(
            model=groundingdino,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        if len(boxes) == 0:
            logger.warning("No walls detected in the image")
            # Save original image if no walls detected
            Image.open(image_path).save(output_path)
            return output_path
        
        # Setup SAM
        sam_predictor.set_image(image_source)
        
        # Transform boxes
        boxes = boxes.to(self.device)
        H, W, _ = image_source.shape
        scaling_tensor = torch.tensor([W, H, W, H], device=self.device)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * scaling_tensor
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        )
        
        # Generate masks
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # Apply color
        color_rgb = Config.get_color_rgb(color_name)
        result_image = self._apply_color_to_mask(
            image_source, masks[0][0].cpu().numpy(), color_rgb
        )
        
        # Save result
        result_pil = Image.fromarray(result_image)
        result_pil.save(output_path)
        
        logger.info(f"Wall color changed to {color_name}")
        return output_path
    
    def _apply_color_to_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Apply color to masked region with blending."""
        h, w = mask.shape[-2:]
        color_layer = np.zeros((h, w, 3), dtype=np.uint8)
        color_layer[:] = color
        
        mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
        
        # Blend with original image
        blended = image.astype(np.float32) * (1 - mask_3d * 0.7) + \
                  color_layer.astype(np.float32) * (mask_3d * 0.7)
        
        return blended.astype(np.uint8)
    
    def detect_objects(
        self,
        image_path: str,
        objects_to_detect: Optional[List[str]] = None,
        box_threshold: float = None,
        text_threshold: float = None
    ) -> List[dict]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to input image
            objects_to_detect: List of objects to detect (uses defaults if None)
            box_threshold: Detection box threshold
            text_threshold: Detection text threshold
            
        Returns:
            List of detected objects with confidence scores
        """
        box_threshold = box_threshold or Config.BOX_THRESHOLD
        text_threshold = text_threshold or Config.TEXT_THRESHOLD
        
        if objects_to_detect is None:
            objects_to_detect = Config.DEFAULT_DETECTABLE_OBJECTS
        
        # Load model
        groundingdino = self._load_grounding_dino()
        
        from groundingdino.util.inference import load_image, predict
        
        # Load image
        image_source, image_tensor = load_image(image_path)
        
        # Create caption
        caption = ", ".join(objects_to_detect)
        
        # Detect
        boxes, logits, phrases = predict(
            model=groundingdino,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Format results
        results = []
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            results.append({
                'index': i + 1,
                'object': phrase,
                'confidence': float(logit),
                'bounding_box': box.tolist()
            })
        
        return results
    
    def apply_wallpaper(
        self,
        room_image_path: str,
        wallpaper_image_path: str,
        output_path: str
    ) -> str:
        """
        Apply wallpaper pattern to walls in an image.
        
        Args:
            room_image_path: Path to room image
            wallpaper_image_path: Path to wallpaper pattern
            output_path: Path to save output image
            
        Returns:
            Path to output image
        """
        # Generate wall mask
        mask = self._generate_wall_mask(room_image_path)
        
        if mask is None:
            raise ValueError("No walls detected in the image")
        
        # Overlay wallpaper
        result = self._overlay_wallpaper(room_image_path, wallpaper_image_path, mask)
        
        # Save result
        cv2.imwrite(output_path, result)
        
        logger.info("Wallpaper applied successfully")
        return output_path
    
    def _generate_wall_mask(self, image_path: str) -> Optional[np.ndarray]:
        """Generate mask for walls in the image."""
        # Load models
        groundingdino = self._load_grounding_dino()
        sam_predictor = self._load_sam()
        
        from groundingdino.util import box_ops
        from groundingdino.util.inference import load_image, predict
        
        # Load image
        image_source, image_tensor = load_image(image_path)
        
        # Detect walls
        boxes, logits, phrases = predict(
            model=groundingdino,
            image=image_tensor,
            caption="wall",
            box_threshold=Config.BOX_THRESHOLD,
            text_threshold=Config.TEXT_THRESHOLD
        )
        
        if len(boxes) == 0:
            return None
        
        # Setup SAM
        sam_predictor.set_image(image_source)
        
        # Transform boxes
        boxes = boxes.to(self.device)
        H, W, _ = image_source.shape
        scaling_tensor = torch.tensor([W, H, W, H], device=self.device)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * scaling_tensor
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        )
        
        # Generate masks
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # Combine all wall masks
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask[0].cpu().numpy().astype(np.uint8))
        
        return combined_mask * 255
    
    def _overlay_wallpaper(
        self,
        room_image_path: str,
        wallpaper_image_path: str,
        mask: np.ndarray
    ) -> np.ndarray:
        """Overlay wallpaper pattern on walls."""
        # Load images
        room_image = cv2.imread(room_image_path)
        wallpaper = cv2.imread(wallpaper_image_path)
        
        # Resize wallpaper to tile across the image
        h, w = room_image.shape[:2]
        
        # Create tiled wallpaper
        wp_h, wp_w = wallpaper.shape[:2]
        tiles_x = (w // wp_w) + 1
        tiles_y = (h // wp_h) + 1
        
        tiled_wallpaper = np.tile(wallpaper, (tiles_y, tiles_x, 1))
        tiled_wallpaper = tiled_wallpaper[:h, :w]
        
        # Apply mask
        mask_3d = np.stack([mask / 255.0] * 3, axis=-1)
        
        # Blend
        result = room_image.astype(np.float32) * (1 - mask_3d) + \
                 tiled_wallpaper.astype(np.float32) * mask_3d
        
        return result.astype(np.uint8)

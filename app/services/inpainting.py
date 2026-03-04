"""
AI Inpainting Service.

This module provides Stable Diffusion-based inpainting for
object replacement in interior images.
"""

import os
import logging
import numpy as np
import torch
from PIL import Image
from typing import Optional

from ..config import Config

logger = logging.getLogger(__name__)


class InpaintingService:
    """Service for AI-powered image inpainting."""
    
    _instance = None
    _pipeline = None
    
    def __new__(cls):
        """Singleton pattern for efficient model loading."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the inpainting service."""
        self.device = Config.DEVICE
        logger.info(f"Inpainting service initialized on device: {self.device}")
    
    def _load_pipeline(self):
        """Load Stable Diffusion inpainting pipeline lazily."""
        if InpaintingService._pipeline is not None:
            return InpaintingService._pipeline
        
        try:
            # Prefer the auto pipeline to support multiple repo/pipeline layouts.
            try:
                from diffusers import AutoPipelineForInpainting as _InpaintPipeline
            except Exception:  # pragma: no cover
                from diffusers import StableDiffusionInpaintPipeline as _InpaintPipeline
            from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
            
            dtype = torch.float16 if Config.USE_GPU else torch.float32

            pretrained_id = Config.SD_INPAINTING_MODEL
            token = Config.HF_TOKEN

            common_kwargs = {
                "torch_dtype": dtype,
                "safety_checker": None,
                "cache_dir": Config.MODEL_CACHE_DIR,
            }
            if token:
                common_kwargs["token"] = token
            
            try:
                pipeline = _InpaintPipeline.from_pretrained(
                    pretrained_id,
                    **common_kwargs,
                ).to(self.device)
            except TypeError:
                # Backward compatibility for older diffusers/huggingface_hub
                if "token" in common_kwargs:
                    common_kwargs["use_auth_token"] = common_kwargs.pop("token")
                pipeline = _InpaintPipeline.from_pretrained(
                    pretrained_id,
                    **common_kwargs,
                ).to(self.device)
            
            # Enable memory optimization if available
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
            
            InpaintingService._pipeline = pipeline
            logger.info("Stable Diffusion inpainting pipeline loaded")
            return pipeline

        except (RepositoryNotFoundError, GatedRepoError) as e:
            # Hugging Face often returns 404 for gated repos when unauthenticated.
            model_id = Config.SD_INPAINTING_MODEL
            msg = (
                f"Cannot access inpainting model '{model_id}'. This model may be gated/private. "
                "Fix: 1) Create/login to Hugging Face, 2) Visit the model page and accept its license/terms, "
                "3) Create an access token, and 4) set HF_TOKEN in your .env. "
                "Alternatively set SD_INPAINTING_MODEL to a public inpainting model. "
                f"Original error: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e
            
        except Exception as e:
            # diffusers often wraps Hub/auth issues as a generic OSError.
            def _iter_exception_chain(ex: BaseException):
                seen = set()
                cur = ex
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    yield cur
                    cur = cur.__cause__ or cur.__context__

            chain_text = "\n".join(str(x) for x in _iter_exception_chain(e))
            message = str(e)

            if any(isinstance(x, (RepositoryNotFoundError, GatedRepoError)) for x in _iter_exception_chain(e)) or (
                "Repository Not Found" in chain_text
                or "gated" in chain_text.lower()
                or "401" in chain_text
                or "403" in chain_text
                or "not authorized" in chain_text.lower()
                or "private" in chain_text.lower()
                or "fetch metadata from the Hub" in chain_text
            ):
                model_id = Config.SD_INPAINTING_MODEL
                help_msg = (
                    f"Cannot download inpainting model '{model_id}' from Hugging Face. "
                    "This is usually because the model is gated/private and Hugging Face hides it as a 404 unless you are authenticated. "
                    "Fix options: (A) set HF_TOKEN in your .env after accepting the model license on Hugging Face, or "
                    "(B) set SD_INPAINTING_MODEL in your .env to a public inpainting model. "
                    f"Original error: {message}"
                )
                logger.error(help_msg)
                raise RuntimeError(help_msg) from e

            logger.error(f"Failed to load inpainting pipeline: {e}")
            raise
    
    def _load_segmentation_service(self):
        """Load segmentation service for object detection."""
        from .segmentation import SegmentationService
        return SegmentationService()
    
    def inpaint(
        self,
        image_path: str,
        object_to_detect: str,
        inpaint_prompt: str,
        output_path: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> str:
        """
        Detect an object and replace it using AI inpainting.
        
        Args:
            image_path: Path to input image
            object_to_detect: Object to find and replace
            inpaint_prompt: Description of the replacement
            output_path: Path to save output image
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Path to output image
        """
        logger.info(f"Inpainting: replacing '{object_to_detect}' with '{inpaint_prompt}'")
        
        # Generate mask for the object
        mask = self._generate_object_mask(image_path, object_to_detect)
        
        if mask is None:
            raise ValueError(f"Object '{object_to_detect}' not found in the image")
        
        # Load pipeline
        pipeline = self._load_pipeline()
        
        # Load and resize images
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize to 512x512 for inpainting
        image_resized = image.resize((512, 512), Image.Resampling.LANCZOS)
        mask_resized = Image.fromarray(mask).resize((512, 512), Image.Resampling.NEAREST)
        
        # Run inpainting
        result = pipeline(
            prompt=inpaint_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Resize back to original size
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        # Save result
        result.save(output_path)
        
        logger.info(f"Inpainting completed: {output_path}")
        return output_path
    
    def _generate_object_mask(
        self,
        image_path: str,
        object_to_detect: str
    ) -> Optional[np.ndarray]:
        """Generate mask for the specified object."""
        try:
            from .segmentation import SegmentationService
            from groundingdino.util import box_ops
            from groundingdino.util.inference import load_image, predict
            
            # Get segmentation service
            seg_service = SegmentationService()
            groundingdino = seg_service._load_grounding_dino()
            sam_predictor = seg_service._load_sam()
            
            # Load image
            image_source, image_tensor = load_image(image_path)
            
            # Detect object
            boxes, logits, phrases = predict(
                model=groundingdino,
                image=image_tensor,
                caption=object_to_detect,
                box_threshold=Config.BOX_THRESHOLD,
                text_threshold=Config.TEXT_THRESHOLD
            )
            
            if len(boxes) == 0:
                return None
            
            # Use first detected box
            box = boxes[0]
            
            # Setup SAM
            sam_predictor.set_image(image_source)
            
            # Transform box
            H, W, _ = image_source.shape
            device = seg_service.device
            box = box.unsqueeze(0).to(device)
            scaling_tensor = torch.tensor([W, H, W, H], device=device)
            box_xyxy = box_ops.box_cxcywh_to_xyxy(box) * scaling_tensor
            transformed_box = sam_predictor.transform.apply_boxes_torch(
                box_xyxy, image_source.shape[:2]
            )
            
            # Generate mask
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False
            )
            
            mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255
            return mask
            
        except Exception as e:
            logger.error(f"Failed to generate object mask: {e}")
            return None
    
    def inpaint_with_mask(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Inpaint image using provided mask.
        
        Args:
            image: Input PIL Image
            mask: Mask PIL Image (white = area to inpaint)
            prompt: Description of the content to generate
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Inpainted PIL Image
        """
        pipeline = self._load_pipeline()
        
        # Resize to 512x512
        original_size = image.size
        image_resized = image.resize((512, 512), Image.Resampling.LANCZOS)
        mask_resized = mask.resize((512, 512), Image.Resampling.NEAREST)
        
        # Run inpainting
        result = pipeline(
            prompt=prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Resize back
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        return result

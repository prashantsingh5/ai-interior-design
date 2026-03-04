"""
Neural Style Transfer Service.

This module provides VGG19-based neural style transfer for
applying artistic styles to interior images.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from typing import Tuple

from ..config import Config

logger = logging.getLogger(__name__)


class ContentLoss(nn.Module):
    """Content loss for style transfer."""
    
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """Style loss using Gram matrix."""
    
    def __init__(self, target):
        super().__init__()
        self.target = self._gram_matrix(target).detach()
        self.loss = 0
    
    def _gram_matrix(self, input):
        batch, channels, height, width = input.size()
        features = input.view(batch * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch * channels * height * width)
    
    def forward(self, input):
        gram = self._gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input


class Normalization(nn.Module):
    """Normalize image with ImageNet mean and std."""
    
    def __init__(self, device):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
        self.mean = mean
        self.std = std
    
    def forward(self, img):
        return (img - self.mean) / self.std


class StyleTransferService:
    """Service for neural style transfer operations."""
    
    def __init__(self):
        """Initialize the style transfer service."""
        self.device = Config.DEVICE
        self.image_size = 512
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        logger.info(f"Style transfer service initialized on device: {self.device}")
    
    def _load_image(self, image_bytes: bytes) -> torch.Tensor:
        """Load and preprocess image from bytes."""
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Resize while maintaining aspect ratio
        max_size = self.image_size
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def _get_model_and_losses(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor
    ) -> Tuple[nn.Sequential, list, list]:
        """Build the style transfer model with loss layers."""
        
        # Load VGG19
        from torchvision.models import VGG19_Weights
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.device).eval()
        
        # Normalization
        normalization = Normalization(self.device).to(self.device)
        
        content_losses = []
        style_losses = []
        
        model = nn.Sequential(normalization)
        
        i = 0  # Increment for conv layers
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                continue
            
            model.add_module(name, layer)
            
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)
            
            if name in self.style_layers:
                target = model(style_img).detach()
                style_loss = StyleLoss(target)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)
        
        # Trim layers after the last loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (ContentLoss, StyleLoss)):
                break
        
        model = model[:i + 1]
        
        return model, content_losses, style_losses
    
    def transfer_style(
        self,
        content_image: bytes,
        style_image: bytes,
        epochs: int = 60,
        style_weight: float = 1e6,
        content_weight: float = 1
    ) -> Image.Image:
        """
        Apply style transfer to content image.
        
        Args:
            content_image: Content image bytes
            style_image: Style image bytes
            epochs: Number of optimization iterations
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            
        Returns:
            Stylized PIL Image
        """
        logger.info("Starting style transfer...")
        
        # Load images
        content_img = self._load_image(content_image)
        style_img = self._load_image(style_image)
        
        # Ensure same size
        style_img = F.interpolate(
            style_img, 
            size=content_img.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Initialize input as content image
        input_img = content_img.clone()
        input_img.requires_grad_(True)
        
        # Build model
        model, content_losses, style_losses = self._get_model_and_losses(
            content_img, style_img
        )
        
        # Optimizer
        optimizer = optim.LBFGS([input_img])
        
        run = [0]
        while run[0] <= epochs:
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 10 == 0:
                    logger.debug(
                        f"Epoch {run[0]}: Style={style_score.item():.4f}, "
                        f"Content={content_score.item():.4f}"
                    )
                
                return style_score + content_score
            
            optimizer.step(closure)
        
        # Clamp and convert to PIL
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        output = input_img.cpu().squeeze(0)
        output_image = transforms.ToPILImage()(output)
        
        logger.info("Style transfer completed")
        return output_image

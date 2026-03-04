"""
Image Recommendation Service.

This module provides visual similarity-based recommendations
for interior design images using ResNet50 feature extraction.
"""

import os
import logging
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional

# Handle optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..config import Config

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for image similarity and recommendations."""
    
    def __init__(self):
        """Initialize the recommendation service."""
        self.model = None
        self.features_df = None
        self._tensorflow_available = False
        self._load_model()
        self._load_features()
        
        logger.info("Recommendation service initialized")
    
    def _load_model(self):
        """Load ResNet50 model for feature extraction."""
        try:
            from tensorflow.keras.applications.resnet50 import ResNet50
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
            from tensorflow.keras.models import Model
            
            # Load base model
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Add layers for feature extraction
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            
            self.model = Model(inputs=base_model.input, outputs=x)
            self._tensorflow_available = True
            
            logger.info("ResNet50 model loaded for feature extraction")
            
        except ImportError as e:
            logger.warning(f"TensorFlow not available: {e}")
            self._tensorflow_available = False
        except Exception as e:
            logger.error(f"Failed to load ResNet50: {e}")
            self._tensorflow_available = False
    
    def _load_features(self):
        """Load precomputed image features."""
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available, skipping features loading")
            self.features_df = None
            return
            
        try:
            features_path = os.path.join(
                Config.DATA_FOLDER,
                'image_features.csv'
            )
            
            if os.path.exists(features_path):
                self.features_df = pd.read_csv(features_path)
                logger.info(f"Loaded {len(self.features_df)} precomputed features")
            else:
                self.features_df = None
                logger.warning(
                    f"Features file not found at {features_path}. "
                    "Recommendations may be limited."
                )
                
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            self.features_df = None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        # Resize to 224x224
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Preprocess for ResNet
        image = preprocess_input(image.astype(np.float32))
        
        return image
    
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract feature embedding from image."""
        if not self._tensorflow_available or self.model is None:
            # Return a simple feature if model not available
            return np.mean(image, axis=(0, 1)).flatten()
            
        preprocessed = self._preprocess_image(image)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        embedding = self.model.predict(preprocessed, verbose=0)
        return embedding.flatten()
    
    def extract_dominant_color(self, image: np.ndarray, n_colors: int = 3) -> List[int]:
        """Extract dominant color from image using K-means clustering."""
        if not SKLEARN_AVAILABLE:
            # Simple fallback: return average color
            avg_color = np.mean(image.reshape(-1, 3), axis=0)
            return [int(c) for c in avg_color]
        
        from sklearn.cluster import KMeans
        
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the most common cluster center
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        dominant_color = kmeans.cluster_centers_[dominant_idx]
        
        return [int(c) for c in dominant_color]
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast using standard deviation."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return float(np.std(gray))
    
    def color_similarity(self, color1: List[int], color2: List[int]) -> float:
        """Calculate color similarity using Delta E (CIE2000)."""
        # Simple Euclidean distance in RGB space (simplified version)
        diff = np.array(color1) - np.array(color2)
        return 1 / (1 + np.sqrt(np.sum(diff ** 2)))
    
    def get_recommendations(
        self,
        image_file,
        category: str = 'Bedroom',
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get similar image recommendations.
        
        Args:
            image_file: Uploaded image file
            category: Room category to filter by
            limit: Number of recommendations to return
            
        Returns:
            Dict with recommendations and analysis
        """
        # Read and preprocess uploaded image
        img = Image.open(image_file.stream if hasattr(image_file, 'stream') else image_file)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Extract features
        embedding = self.get_embedding(img_array)
        dominant_color = self.extract_dominant_color(img_array)
        contrast = self.calculate_contrast(img_array)
        
        recommendations = []
        
        if self.features_df is not None:
            # Filter by category if features exist
            filtered_df = self.features_df
            if 'category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['category'] == category]
            
            if len(filtered_df) > 0:
                # Calculate similarity
                feature_cols = [col for col in filtered_df.columns if col.startswith('feature_')]
                
                if feature_cols:
                    stored_embeddings = filtered_df[feature_cols].values
                    similarities = cosine_similarity([embedding], stored_embeddings)[0]
                    
                    # Get top recommendations
                    top_indices = np.argsort(similarities)[::-1][:limit]
                    
                    for idx in top_indices:
                        row = filtered_df.iloc[idx]
                        recommendations.append({
                            'path': row.get('path', f'image_{idx}'),
                            'similarity': float(similarities[idx]),
                            'category': row.get('category', category)
                        })
        
        return {
            'recommendations': recommendations,
            'analysis': {
                'dominant_color': dominant_color,
                'contrast': contrast,
                'category': category
            },
            'total': len(recommendations)
        }
    
    def analyze_image(self, image_file) -> Dict[str, Any]:
        """
        Analyze an interior image for visual characteristics.
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            Dict with color analysis and features
        """
        # Read image
        img = Image.open(image_file.stream if hasattr(image_file, 'stream') else image_file)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        # Resize for analysis
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Extract features
        dominant_color = self.extract_dominant_color(img_resized)
        contrast = self.calculate_contrast(img_resized)
        
        # Get multiple colors
        colors = self._get_color_palette(img_resized, n_colors=5)
        
        # Brightness analysis
        brightness = self._calculate_brightness(img_resized)
        
        return {
            'dominant_color': {
                'rgb': dominant_color,
                'hex': '#{:02x}{:02x}{:02x}'.format(*dominant_color)
            },
            'color_palette': [
                {
                    'rgb': color,
                    'hex': '#{:02x}{:02x}{:02x}'.format(*color)
                }
                for color in colors
            ],
            'contrast': contrast,
            'brightness': brightness,
            'style_indicators': {
                'is_bright': brightness > 128,
                'is_high_contrast': contrast > 50,
                'color_mood': self._get_color_mood(dominant_color)
            }
        }
    
    def _get_color_palette(self, image: np.ndarray, n_colors: int = 5) -> List[List[int]]:
        """Extract color palette from image."""
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = []
        for center in kmeans.cluster_centers_:
            colors.append([int(c) for c in center])
        
        return colors
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray))
    
    def _get_color_mood(self, color: List[int]) -> str:
        """Determine color mood/tone."""
        r, g, b = color
        
        # Simple mood classification based on dominant channel
        if r > g and r > b:
            return 'warm'
        elif b > r and b > g:
            return 'cool'
        elif g > r and g > b:
            return 'natural'
        elif abs(r - g) < 30 and abs(g - b) < 30:
            return 'neutral'
        else:
            return 'balanced'

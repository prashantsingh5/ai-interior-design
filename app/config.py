"""
Configuration management for AI Interior Design Studio.
"""

import os
from dotenv import load_dotenv

# Handle optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

load_dotenv()


def _check_cuda_available():
    """Check if CUDA is available."""
    if TORCH_AVAILABLE:
        return torch.cuda.is_available()
    return False


class Config:
    """Application configuration class."""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    
    # Directory Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'media')
    WEIGHTS_FOLDER = os.path.join(BASE_DIR, 'weights')
    DATA_FOLDER = os.path.join(BASE_DIR, 'data')
    
    # Upload Configuration
    MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', 10))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Convert to bytes
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # GPU Configuration
    USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true' and _check_cuda_available()
    DEVICE = 'cuda' if USE_GPU else 'cpu'
    
    # Model Configuration
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', os.path.join(BASE_DIR, 'weights'))

    # Hugging Face Hub auth (needed for gated/private models)
    HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
    
    # GroundingDINO Configuration
    GROUNDING_DINO_REPO = "ShilongLiu/GroundingDINO"
    GROUNDING_DINO_SWINB_CKPT = "groundingdino_swinb_cogcoor.pth"
    GROUNDING_DINO_SWINB_CONFIG = "GroundingDINO_SwinB.cfg.py"
    GROUNDING_DINO_SWINT_CKPT = "groundingdino_swint_ogc.pth"
    GROUNDING_DINO_SWINT_CONFIG = "GroundingDINO_SwinT_OGC.cfg.py"
    
    # SAM Configuration
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    
    # Stable Diffusion Configuration
    # Default to a public inpainting model to avoid gated/private Hub access.
    SD_INPAINTING_MODEL = os.getenv('SD_INPAINTING_MODEL', "stable-diffusion-v1-5/stable-diffusion-inpainting")
    
    # Detection Thresholds
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25
    
    # Color Map for Wall Color Change
    COLOR_MAP = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "light red": (255, 102, 102),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "light blue": (173, 216, 230),
        "pink": (255, 192, 203),
        "brown": (165, 42, 42),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "lime": (0, 255, 0),
        "maroon": (128, 0, 0),
        "navy": (0, 0, 128),
        "olive": (128, 128, 0),
        "teal": (0, 128, 128),
        "silver": (192, 192, 192),
        "beige": (245, 245, 220),
        "cream": (255, 253, 208),
        "lavender": (230, 230, 250),
        "coral": (255, 127, 80),
        "salmon": (250, 128, 114),
        "gold": (255, 215, 0),
        "ivory": (255, 255, 240),
        "mint": (189, 252, 201),
        "peach": (255, 218, 185),
        "terracotta": (204, 78, 92),
        "sage": (188, 184, 138),
        "charcoal": (54, 69, 79),
        "slate": (112, 128, 144),
        "burgundy": (128, 0, 32),
        "forest green": (34, 139, 34),
        "sky blue": (135, 206, 235),
        "powder blue": (176, 224, 230),
        "dusty rose": (201, 111, 131),
        "champagne": (247, 231, 206)
    }
    
    # Default Objects for Detection
    DEFAULT_DETECTABLE_OBJECTS = [
        "bed frame", "headboard", "footboard", "dresser", "nightstand", "desk", "chair",
        "bedside table", "dressing table", "wardrobe", "closet organizer", "bookshelf",
        "ottoman", "storage bench", "accent chair", "table lamp", "floor lamp",
        "ceiling light", "string lights", "sconce", "artwork", "picture", "mirror",
        "rug", "curtain", "blinds", "plant", "candle", "basket",
        "closet", "drawer", "storage bin", "TV", "computer", "phone", "tablet",
        "speaker", "bed", "wall", "window", "ceiling", "floor", "sofa", "couch",
        "coffee table", "dining table", "cabinet", "shelf", "lamp", "vase"
    ]
    
    @classmethod
    def get_color_rgb(cls, color_name: str) -> tuple:
        """Get RGB color value from color name."""
        return cls.COLOR_MAP.get(color_name.lower(), (128, 128, 128))
    
    @classmethod
    def is_allowed_file(cls, filename: str) -> bool:
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    FLASK_ENV = 'production'


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development')
    return config_map.get(env, DevelopmentConfig)

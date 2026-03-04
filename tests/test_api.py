"""
API Tests for AI Interior Design Studio.
"""

import os
import sys
import pytest
import io
from PIL import Image

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.config import TestingConfig


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app(TestingConfig)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (224, 224), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoints:
    """Test health and info endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'version' in data
    
    def test_index(self, client):
        """Test index endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        data = response.get_json()
        assert 'endpoints' in data
        assert 'message' in data


class TestWallColorAPI:
    """Test wall color change API."""
    
    def test_get_colors(self, client):
        """Test get available colors endpoint."""
        response = client.get('/api/wall-color/colors')
        assert response.status_code == 200
        data = response.get_json()
        assert 'colors' in data
        assert len(data['colors']) > 0
    
    def test_change_color_no_image(self, client):
        """Test color change without image."""
        response = client.post('/api/wall-color/change')
        assert response.status_code == 400


class TestStyleTransferAPI:
    """Test style transfer API."""
    
    def test_get_info(self, client):
        """Test style transfer info endpoint."""
        response = client.get('/api/style-transfer/info')
        assert response.status_code == 200
        data = response.get_json()
        assert 'model' in data
        assert data['model'] == 'VGG19'


class TestObjectDetectionAPI:
    """Test object detection API."""
    
    def test_get_default_objects(self, client):
        """Test get default objects endpoint."""
        response = client.get('/api/objects/default-objects')
        assert response.status_code == 200
        data = response.get_json()
        assert 'objects' in data
        assert len(data['objects']) > 0


class TestInpaintingAPI:
    """Test inpainting API."""
    
    def test_get_info(self, client):
        """Test inpainting info endpoint."""
        response = client.get('/api/inpaint/info')
        assert response.status_code == 200
        data = response.get_json()
        assert 'service' in data
        assert 'example_prompts' in data


class TestRecommendationsAPI:
    """Test recommendations API."""
    
    def test_get_categories(self, client):
        """Test get categories endpoint."""
        response = client.get('/api/recommendations/categories')
        assert response.status_code == 200
        data = response.get_json()
        assert 'categories' in data


class TestWallpaperAPI:
    """Test wallpaper API."""
    
    def test_get_info(self, client):
        """Test wallpaper info endpoint."""
        response = client.get('/api/wallpaper/info')
        assert response.status_code == 200
        data = response.get_json()
        assert 'service' in data


class TestTilesAPI:
    """Test tiles API."""
    
    def test_get_info(self, client):
        """Test tiles info endpoint."""
        response = client.get('/api/tiles/info')
        assert response.status_code == 200
        data = response.get_json()
        assert 'service' in data
    
    def test_get_gallery(self, client):
        """Test tiles gallery endpoint."""
        response = client.get('/api/tiles/gallery')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

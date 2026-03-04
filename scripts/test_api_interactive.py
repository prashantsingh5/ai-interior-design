"""
Interactive API Test Script
Test the AI Interior Design features step by step
"""
import requests
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("HEALTH CHECK")
    print("="*50)
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"Version: {data['version']}")
    print(f"GPU Available: {data['gpu_available']}")
    return data['status'] == 'healthy'

def test_available_colors():
    """Get available wall colors"""
    print("\n" + "="*50)
    print("AVAILABLE COLORS")
    print("="*50)
    response = requests.get(f"{BASE_URL}/api/wall-color/colors")
    data = response.json()
    colors = data.get('colors', [])
    total = data.get('total', len(colors))
    print(f"Found {total} colors:")
    for i, color in enumerate(colors[:10]):
        name = color.get('name', 'unknown')
        rgb = color.get('rgb', [0, 0, 0])
        print(f"  {i+1}. {name}: RGB{tuple(rgb)}")
    if total > 10:
        print(f"  ... and {total - 10} more")
    return colors

def test_wall_color_change(image_path, color="sky blue", output_path="output_wall.png"):
    """Test wall color change using multipart form data"""
    print("\n" + "="*50)
    print("WALL COLOR CHANGE")
    print("="*50)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    print(f"Input: {image_path}")
    print(f"Color: {color}")
    
    # Send multipart form data
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {'text_prompt': 'wall', 'color_name': color}
        
        print("Sending request...")
        response = requests.post(f"{BASE_URL}/api/wall-color/change", files=files, data=data)
    
    if response.status_code == 200:
        # Save binary image response
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"SUCCESS! Saved: {output_path}")
        return True
    else:
        try:
            error = response.json()
            print(f"Error: {error.get('error', 'Unknown error')}")
        except:
            print(f"HTTP {response.status_code}: {response.text[:200]}")
        return False

def test_object_detection(image_path, prompt="wall"):
    """Test object detection using multipart form data"""
    print("\n" + "="*50)
    print("OBJECT DETECTION")
    print("="*50)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    print(f"Input: {image_path}")
    print(f"Prompt: {prompt}")
    
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {'prompt': prompt}
        
        print("Sending request...")
        response = requests.post(f"{BASE_URL}/api/objects/detect", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        boxes = result.get('boxes', [])
        print(f"Detected {len(boxes)} objects")
        return True
    else:
        try:
            error = response.json()
            print(f"Error: {error.get('error', 'Unknown error')}")
        except:
            print(f"HTTP {response.status_code}: {response.text[:200]}")
        return False

def main():
    print("=" * 50)
    print("   AI INTERIOR DESIGN - API TEST SUITE")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("Server not healthy. Exiting.")
        return
    
    # Show available colors
    colors = test_available_colors()
    
    # Find test image
    test_images = [
        "data/test/test_img.jpg",
        "../final/media/1/input/test_img.jpg",
        "test_image.jpg"
    ]
    
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if test_image:
        print(f"\nFound test image: {test_image}")
        
        # Test wall color change
        print("\n" + "-"*50)
        print("Testing Wall Color Change...")
        test_wall_color_change(test_image, "coral", "output_wall_coral.png")
        
        print("\n" + "-"*50)
        print("Testing Object Detection...")
        test_object_detection(test_image, "wall")
    else:
        print("\nNo test image found!")
        print("Copy an image to 'data/test/test_img.jpg' or 'test_image.jpg'")
    
    print("\n" + "="*50)
    print("Test complete!")
    print("="*50)

if __name__ == "__main__":
    main()

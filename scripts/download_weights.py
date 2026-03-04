"""
Model Weights Download Script.

This script downloads required model weights for the AI Interior Design Studio.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_sam_weights(weights_dir: str):
    """Download SAM weights from Meta AI."""
    import requests
    from tqdm import tqdm
    
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    sam_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
    
    if os.path.exists(sam_path):
        print(f"✓ SAM weights already exist at {sam_path}")
        return True
    
    print(f"Downloading SAM weights...")
    print(f"  URL: {sam_url}")
    print(f"  Destination: {sam_path}")
    
    try:
        response = requests.get(sam_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(sam_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc="SAM Weights") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ SAM weights downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading SAM weights: {e}")
        return False


def download_grounding_dino(weights_dir: str):
    """Download GroundingDINO weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    
    repo_id = "ShilongLiu/GroundingDINO"
    files = [
        "groundingdino_swinb_cogcoor.pth",
        "GroundingDINO_SwinB.cfg.py",
        "groundingdino_swint_ogc.pth",
        "GroundingDINO_SwinT_OGC.cfg.py"
    ]
    
    print("Downloading GroundingDINO weights from HuggingFace...")
    
    for filename in files:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=os.path.join(weights_dir, 'huggingface')
            )
            print(f"✓ {filename}")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False
    
    return True


def install_grounding_dino():
    """Install GroundingDINO from GitHub."""
    import subprocess
    
    print("\nInstalling GroundingDINO...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/IDEA-Research/GroundingDINO.git"
        ], check=True)
        print("✓ GroundingDINO installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing GroundingDINO: {e}")
        print("  Please install manually: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        return False


def setup_stable_diffusion():
    """Pre-download Stable Diffusion weights."""
    print("\nPre-caching Stable Diffusion inpainting model...")
    
    try:
        try:
            from diffusers import AutoPipelineForInpainting as _InpaintPipeline
        except Exception:
            from diffusers import StableDiffusionInpaintPipeline as _InpaintPipeline
        import torch
        
        model_id = os.getenv("SD_INPAINTING_MODEL", "stable-diffusion-v1-5/stable-diffusion-inpainting")
        print(f"  Downloading {model_id}...")
        print("  This may take a while (several GB)...")
        
        # Just load to cache, don't keep in memory
        pipe = _InpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        del pipe
        
        print("✓ Stable Diffusion inpainting model cached")
        return True
        
    except Exception as e:
        print(f"✗ Error caching Stable Diffusion: {e}")
        print("  The model will be downloaded on first use")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download model weights for AI Interior Design Studio'
    )
    parser.add_argument(
        '--weights-dir',
        type=str,
        default='weights',
        help='Directory to store weights'
    )
    parser.add_argument(
        '--skip-sam',
        action='store_true',
        help='Skip SAM weights download'
    )
    parser.add_argument(
        '--skip-grounding-dino',
        action='store_true',
        help='Skip GroundingDINO setup'
    )
    parser.add_argument(
        '--skip-stable-diffusion',
        action='store_true',
        help='Skip Stable Diffusion pre-caching'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI Interior Design Studio - Model Setup")
    print("=" * 60)
    
    # Create weights directory
    weights_dir = os.path.abspath(args.weights_dir)
    os.makedirs(weights_dir, exist_ok=True)
    print(f"\nWeights directory: {weights_dir}\n")
    
    success = True
    
    # Download SAM
    if not args.skip_sam:
        if not download_sam_weights(weights_dir):
            success = False
    
    # Setup GroundingDINO
    if not args.skip_grounding_dino:
        if not download_grounding_dino(weights_dir):
            success = False
        if not install_grounding_dino():
            success = False
    
    # Pre-cache Stable Diffusion
    if not args.skip_stable_diffusion:
        setup_stable_diffusion()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Setup completed successfully!")
        print("\nYou can now run the application with:")
        print("  python run.py")
    else:
        print("⚠ Setup completed with some warnings")
        print("Please check the errors above")
    print("=" * 60)


if __name__ == '__main__':
    main()

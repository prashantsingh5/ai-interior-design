"""
AI Interior Design - Gradio Web Interface
A visual interface for testing all AI features.
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import os
import sys
import tempfile
from io import BytesIO

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import Config

# Global service instances (lazy loaded)
_segmentation_service = None
_style_transfer_service = None
_inpainting_service = None


def get_segmentation_service():
    """Lazy load segmentation service."""
    global _segmentation_service
    if _segmentation_service is None:
        try:
            from app.services.segmentation import SegmentationService
            _segmentation_service = SegmentationService()
        except Exception as e:
            print(f"Could not load SegmentationService: {e}")
            return None
    return _segmentation_service


def get_style_transfer_service():
    """Lazy load style transfer service."""
    global _style_transfer_service
    if _style_transfer_service is None:
        try:
            from app.services.style_transfer import StyleTransferService
            _style_transfer_service = StyleTransferService()
        except Exception as e:
            print(f"Could not load StyleTransferService: {e}")
            return None
    return _style_transfer_service


def get_inpainting_service():
    """Lazy load inpainting service."""
    global _inpainting_service
    if _inpainting_service is None:
        try:
            from app.services.inpainting import InpaintingService
            _inpainting_service = InpaintingService()
        except Exception as e:
            print(f"Could not load InpaintingService: {e}")
            return None
    return _inpainting_service


def _image_to_temp_path(img: Image.Image, suffix: str = ".png") -> str:
    """Write a PIL image to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    img.save(tmp_path)
    return tmp_path


def _bytes_from_pil(img: Image.Image) -> bytes:
    """Convert a PIL image to PNG bytes."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def change_wall_color(image, color_name, text_prompt="wall"):
    """Change wall color in an image."""
    if image is None:
        return None, "Please upload an image"
    
    service = get_segmentation_service()
    if service is None:
        return None, "Segmentation service not available. Check that SAM weights are downloaded."
    
    try:
        # Save input image temporarily
        input_path = "temp_input.png"
        output_path = "temp_output.png"
        
        # Convert to PIL and save
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(input_path)
        else:
            image.save(input_path)
        
        # Process
        result_path = service.change_wall_color(
            image_path=input_path,
            text_prompt=text_prompt,
            color_name=color_name,
            output_path=output_path
        )
        
        # Load result
        result = Image.open(result_path)
        
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        
        return result, f"Successfully changed wall color to {color_name}!"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def apply_style_transfer(content_image, style_image):
    """Apply neural style transfer."""
    if content_image is None or style_image is None:
        return None, "Please upload both content and style images"
    
    service = get_style_transfer_service()
    if service is None:
        return None, "Style transfer service not available"
    
    try:
        # Convert images to bytes
        def image_to_bytes(img):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            return _bytes_from_pil(img)
        
        content_bytes = image_to_bytes(content_image)
        style_bytes = image_to_bytes(style_image)
        
        # Process - returns PIL Image
        result = service.transfer_style(
            content_image=content_bytes,
            style_image=style_bytes,
            epochs=60
        )
        
        return result, "Style transfer complete!"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def detect_objects_ui(image, custom_objects_text: str):
    """Detect objects using GroundingDINO and return annotated image + JSON."""
    if image is None:
        return None, [], "Please upload an image"

    service = get_segmentation_service()
    if service is None:
        return None, [], "Segmentation service not available"

    temp_input = None
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        temp_input = _image_to_temp_path(image, suffix=".png")

        objects_to_detect = None
        custom_objects_text = (custom_objects_text or "").strip()
        if custom_objects_text:
            objects_to_detect = [x.strip() for x in custom_objects_text.split(",") if x.strip()]

        results = service.detect_objects(image_path=temp_input, objects_to_detect=objects_to_detect)

        # Annotate image (GroundingDINO returns cx,cy,w,h normalized)
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]

        annotated = img_np.copy()
        for item in results:
            try:
                cx, cy, bw, bh = item.get("bounding_box", [0, 0, 0, 0])
                x1 = int(max(0, min(w - 1, (cx - bw / 2) * w)))
                y1 = int(max(0, min(h - 1, (cy - bh / 2) * h)))
                x2 = int(max(0, min(w - 1, (cx + bw / 2) * w)))
                y2 = int(max(0, min(h - 1, (cy + bh / 2) * h)))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{item.get('object', 'obj')} {item.get('confidence', 0):.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                continue

        status = f"Detected {len(results)} objects" if results else "No objects detected"
        return Image.fromarray(annotated), results, status

    except Exception as e:
        return None, [], f"Error: {str(e)}"
    finally:
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass


def inpaint_ui(image, object_to_replace: str, replacement_prompt: str, steps: int, guidance: float):
    """Inpaint an object by name and replacement prompt."""
    if image is None:
        return None, "Please upload an image"

    object_to_replace = (object_to_replace or "").strip()
    replacement_prompt = (replacement_prompt or "").strip()
    if not object_to_replace or not replacement_prompt:
        return None, "Please provide both object_to_replace and replacement_prompt"

    service = get_inpainting_service()
    if service is None:
        return None, "Inpainting service not available"

    temp_input = None
    temp_output = None
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        temp_input = _image_to_temp_path(image, suffix=".png")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

        result_path = service.inpaint(
            image_path=temp_input,
            object_to_detect=object_to_replace,
            inpaint_prompt=replacement_prompt,
            output_path=temp_output,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
        )

        result = Image.open(result_path)
        return result, "Inpainting complete!"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        for p in [temp_input, temp_output]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def wallpaper_ui(room_image, wallpaper_image):
    """Apply a wallpaper pattern to detected walls."""
    if room_image is None or wallpaper_image is None:
        return None, "Please upload both a room image and a wallpaper pattern"

    service = get_segmentation_service()
    if service is None:
        return None, "Segmentation service not available"

    room_path = None
    pattern_path = None
    out_path = None
    try:
        if not isinstance(room_image, Image.Image):
            room_image = Image.fromarray(room_image)
        if not isinstance(wallpaper_image, Image.Image):
            wallpaper_image = Image.fromarray(wallpaper_image)

        room_path = _image_to_temp_path(room_image, suffix=".png")
        pattern_path = _image_to_temp_path(wallpaper_image, suffix=".png")
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name

        result_path = service.apply_wallpaper(
            room_image_path=room_path,
            wallpaper_image_path=pattern_path,
            output_path=out_path,
        )

        result = Image.open(result_path)
        return result, "Wallpaper applied!"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        for p in [room_path, pattern_path, out_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def load_tiles_ui():
    """Load tile images from data/tiles and return for Gallery."""
    tiles_dir = os.path.join(Config.DATA_FOLDER, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    valid_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    items = []
    try:
        for name in sorted(os.listdir(tiles_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in valid_ext:
                continue
            path = os.path.join(tiles_dir, name)
            try:
                img = Image.open(path).convert("RGB")
                items.append((img, name))
            except Exception:
                continue

        if not items:
            return [], "No tiles found. Add images to data/tiles/"
        return items, f"Loaded {len(items)} tiles"
    except Exception as e:
        return [], f"Error: {str(e)}"


def simple_color_overlay(image, color_name):
    """Simple color overlay demo (works without AI models)."""
    if image is None:
        return None, "Please upload an image"
    
    try:
        # Get color RGB
        color_rgb = Config.COLOR_MAP.get(color_name.lower(), (128, 128, 128))
        
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Create color overlay
        overlay = np.full_like(img_array, color_rgb, dtype=np.uint8)
        
        # Blend with original (simple demo)
        alpha = 0.3
        result = cv2.addWeighted(img_array, 1 - alpha, overlay, alpha, 0)
        
        return Image.fromarray(result), f"Applied {color_name} overlay (30% blend)"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# Get available colors for dropdown
COLOR_CHOICES = list(Config.COLOR_MAP.keys())

# Create the Gradio interface
with gr.Blocks(title="AI Interior Design") as demo:
    gr.Markdown("""
    # 🏠 AI Interior Design Studio
    
    Transform your interior spaces with AI-powered visualization tools.
    
    **Features:**
    - **Wall Color Change**: Automatically detect walls and change their color
    - **Style Transfer**: Apply artistic styles to your room photos
    - **Object Detection**: Detect furniture and objects in a room
    - **Inpainting**: Replace an object using Stable Diffusion inpainting
    - **Wallpaper**: Apply a wallpaper pattern to walls
    - **Tiles Gallery**: Browse locally available tiles
    - **Color Preview**: Quick color overlay preview
    """)
    
    with gr.Tabs():
        # Tab 1: Wall Color Change
        with gr.TabItem("🎨 Wall Color Change"):
            gr.Markdown("### Change wall colors using AI segmentation")
            
            with gr.Row():
                with gr.Column():
                    wall_input = gr.Image(label="Upload Room Image", type="pil")
                    wall_color = gr.Dropdown(
                        choices=COLOR_CHOICES,
                        value="light blue",
                        label="Select Wall Color"
                    )
                    wall_prompt = gr.Textbox(
                        value="wall",
                        label="Wall Detection Prompt",
                        info="Describe which wall to detect (e.g., 'wall', 'back wall', 'left wall')"
                    )
                    wall_btn = gr.Button("🎨 Change Wall Color", variant="primary")
                
                with gr.Column():
                    wall_output = gr.Image(label="Result")
                    wall_status = gr.Textbox(label="Status")
            
            wall_btn.click(
                fn=change_wall_color,
                inputs=[wall_input, wall_color, wall_prompt],
                outputs=[wall_output, wall_status]
            )
        
        # Tab 2: Style Transfer
        with gr.TabItem("🖼️ Style Transfer"):
            gr.Markdown("### Apply artistic styles to room images")
            
            with gr.Row():
                with gr.Column():
                    content_input = gr.Image(label="Room Image (Content)", type="pil")
                    style_input = gr.Image(label="Style Image", type="pil")
                    style_btn = gr.Button("🎨 Apply Style", variant="primary")
                
                with gr.Column():
                    style_output = gr.Image(label="Styled Result")
                    style_status = gr.Textbox(label="Status")
            
            style_btn.click(
                fn=apply_style_transfer,
                inputs=[content_input, style_input],
                outputs=[style_output, style_status]
            )
        
        # Tab 3: Quick Color Preview (works without AI)
        with gr.TabItem("✨ Quick Color Preview"):
            gr.Markdown("### Quick color overlay preview (no AI required)")
            
            with gr.Row():
                with gr.Column():
                    preview_input = gr.Image(label="Upload Image", type="pil")
                    preview_color = gr.Dropdown(
                        choices=COLOR_CHOICES,
                        value="beige",
                        label="Select Color"
                    )
                    preview_btn = gr.Button("✨ Preview Color", variant="primary")
                
                with gr.Column():
                    preview_output = gr.Image(label="Preview Result")
                    preview_status = gr.Textbox(label="Status")
            
            preview_btn.click(
                fn=simple_color_overlay,
                inputs=[preview_input, preview_color],
                outputs=[preview_output, preview_status]
            )

        # Tab 4: Object Detection
        with gr.TabItem("🔍 Object Detection"):
            gr.Markdown("### Detect objects/furniture using GroundingDINO")

            with gr.Row():
                with gr.Column():
                    det_input = gr.Image(label="Upload Room Image", type="pil")
                    det_objects = gr.Textbox(
                        value="",
                        label="Custom objects (optional)",
                        info="Comma-separated list (e.g., 'sofa, chair, table'). Leave empty to use defaults."
                    )
                    det_btn = gr.Button("🔍 Detect Objects", variant="primary")

                with gr.Column():
                    det_output = gr.Image(label="Annotated Image")
                    det_json = gr.JSON(label="Detections")
                    det_status = gr.Textbox(label="Status")

            det_btn.click(
                fn=detect_objects_ui,
                inputs=[det_input, det_objects],
                outputs=[det_output, det_json, det_status]
            )

        # Tab 5: Inpainting
        with gr.TabItem("🧩 Inpainting"):
            gr.Markdown("### Replace an object using Stable Diffusion inpainting")
            gr.Markdown(
                "First run may take time (model download). Use a simple object name like `sofa` or `chair`."
            )

            with gr.Row():
                with gr.Column():
                    inp_image = gr.Image(label="Upload Room Image", type="pil")
                    inp_object = gr.Textbox(value="sofa", label="Object to replace")
                    inp_prompt = gr.Textbox(value="modern minimalist sofa", label="Replacement prompt")
                    inp_steps = gr.Slider(10, 60, value=30, step=1, label="Inference steps")
                    inp_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                    inp_btn = gr.Button("🧩 Inpaint", variant="primary")
                with gr.Column():
                    inp_output = gr.Image(label="Result")
                    inp_status = gr.Textbox(label="Status")

            inp_btn.click(
                fn=inpaint_ui,
                inputs=[inp_image, inp_object, inp_prompt, inp_steps, inp_guidance],
                outputs=[inp_output, inp_status],
            )

        # Tab 6: Wallpaper
        with gr.TabItem("🧱 Wallpaper"):
            gr.Markdown("### Apply a wallpaper pattern to detected walls")

            with gr.Row():
                with gr.Column():
                    wp_room = gr.Image(label="Room Image", type="pil")
                    wp_pattern = gr.Image(label="Wallpaper Pattern", type="pil")
                    wp_btn = gr.Button("🧱 Apply Wallpaper", variant="primary")
                with gr.Column():
                    wp_output = gr.Image(label="Result")
                    wp_status = gr.Textbox(label="Status")

            wp_btn.click(
                fn=wallpaper_ui,
                inputs=[wp_room, wp_pattern],
                outputs=[wp_output, wp_status],
            )

        # Tab 7: Tiles Gallery
        with gr.TabItem("🧱 Tiles Gallery"):
            gr.Markdown("### Browse tiles from the local data/tiles folder")
            tiles_btn = gr.Button("🔄 Refresh Tiles", variant="primary")
            tiles_gallery = gr.Gallery(label="Tiles", columns=4, height=360)
            tiles_status = gr.Textbox(label="Status")
            tiles_btn.click(
                fn=load_tiles_ui,
                inputs=[],
                outputs=[tiles_gallery, tiles_status]
            )
        
        # Tab 8: Available Colors
        with gr.TabItem("🎨 Color Palette"):
            gr.Markdown("### Available Colors")
            
            # Create color swatches
            def create_color_swatches():
                swatches = []
                for name, rgb in Config.COLOR_MAP.items():
                    swatch = np.full((50, 100, 3), rgb, dtype=np.uint8)
                    swatches.append((name, swatch))
                return swatches
            
            color_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
            for name, rgb in Config.COLOR_MAP.items():
                hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
                color_html += f"""
                <div style='text-align: center;'>
                    <div style='width: 80px; height: 50px; background-color: {hex_color}; 
                         border-radius: 5px; border: 1px solid #ccc;'></div>
                    <small>{name}</small>
                </div>
                """
            color_html += "</div>"
            
            gr.HTML(color_html)
    
    gr.Markdown("""
    ---
    **Tips:**
    - For best results, use well-lit room photos
    - The wall detection works best on clear, visible walls
    - Style transfer may take a few seconds depending on image size
    
    **Requirements:**
    - SAM model weights in `weights/` folder for wall color change
    - GPU recommended for faster processing
    """)


if __name__ == "__main__":
    print("=" * 50)
    print("  AI Interior Design - Gradio Interface")
    print("=" * 50)
    print()
    
    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: Not available (using CPU)")
    except:
        print("GPU: PyTorch not installed")
    
    # Check SAM weights
    sam_path = os.path.join("weights", "sam_vit_h_4b8939.pth")
    print(f"SAM weights: {'Found' if os.path.exists(sam_path) else 'Not found'}")
    print()
    
    # Launch
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

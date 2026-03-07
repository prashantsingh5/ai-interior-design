<div align="center">

# 🏠 AI Interior Design Studio

### Transform Your Spaces with the Power of AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)










[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<br/>

**A production-ready AI platform combining SAM, GroundingDINO, Stable Diffusion, and Neural Style Transfer for intelligent interior design visualization.**

[🚀 Quick Start](#-quick-start) •
[✨ Features](#-features) •
[📸 Demo](#-demo) •
[📖 Documentation](#-api-documentation) •
[🤝 Contributing](#-contributing)

<br/>

<!-- Add your demo GIF here for maximum impact -->
<!-- <img src="docs/images/demo.gif" alt="Demo" width="800"/> -->

---

</div>

## Demo



https://github.com/user-attachments/assets/7a73b3b4-81e7-48eb-88fe-975339558958


---

## 🌟 Why This Project?

Interior design visualization typically requires expensive software or professional services. This project democratizes AI-powered design tools by combining **5 state-of-the-art models** into a single, easy-to-use platform:

| Challenge | Our Solution |
|-----------|--------------|
| "I want to see my room in a different color" | **AI Wall Color Change** - Automatically detects walls and applies realistic colors |
| "How would this painting style look in my room?" | **Neural Style Transfer** - Apply any artistic style to your interior |
| "What if I had a different sofa?" | **AI Inpainting** - Replace any furniture with AI-generated alternatives |
| "I need design inspiration" | **Smart Recommendations** - Get visually similar design suggestions |
| "Would this wallpaper work?" | **Virtual Wallpaper** - Preview wallpapers with perspective-aware application |

<br/>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎨 Wall Color Visualization
Instantly change wall colors using **Segment Anything Model (SAM)** + **GroundingDINO** for pixel-perfect wall detection.

- 40+ preset colors included
- Custom hex color support
- Precise wall boundary detection
- Works with complex room layouts

</td>
<td width="50%">

### 🖼️ Neural Style Transfer
Transform your interiors with artistic styles using **VGG19-based neural style transfer**.

- Fast GPU-accelerated processing
- Adjustable style intensity
- Preserve room structure
- Works with any style image

</td>
</tr>
<tr>
<td width="50%">

### ✏️ AI Inpainting
Replace furniture using **Stable Diffusion Inpainting** for photorealistic results.

- Text-prompt based generation
- Automatic object detection
- Seamless blending
- High-resolution output

</td>
<td width="50%">

### 🔍 Smart Object Detection
Detect and identify room objects with **GroundingDINO** zero-shot detection.

- 50+ detectable object types
- Bounding box visualization
- Confidence scoring
- Custom object queries

</td>
</tr>
<tr>
<td width="50%">

### 🎭 Wallpaper Application
Virtually apply wallpaper patterns with perspective-aware texturing.

- Automatic wall detection
- Pattern tiling
- Perspective correction
- Multiple wall support

</td>
<td width="50%">

### 🏷️ Design Recommendations
Get intelligent suggestions using **ResNet50** feature extraction.

- Visual similarity matching
- Category-based filtering
- Room type classification
- Style clustering

</td>
</tr>
</table>

<br/>

## 📸 Demo

<div align="center">

### Wall Color Change
| Original | Orange | Cyan |
|:--------:|:------:|:----:|
| <img src="https://github.com/user-attachments/assets/6aa5f4bd-6439-44a6-8ea2-f59fad69dc45" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | <img src="https://github.com/user-attachments/assets/d4a6caa1-5ab8-4c5e-9714-7db83b42a862" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | <img src="https://github.com/user-attachments/assets/de8706b3-818e-4782-884e-38356da5d8ea" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> |

### Style Transfer
| Content | Style | Result |
|:-------:|:-----:|:------:|
| <img src="https://github.com/user-attachments/assets/e4f92119-ef19-4dba-95e7-9e6e049bad2f" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | <img src="https://github.com/user-attachments/assets/84fa4ada-f4d7-45cf-b821-7631b8844fb5" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | <img src="https://github.com/user-attachments/assets/092d9235-cefd-45b0-b028-53b5e3d2d1f1" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> |

### AI Inpainting
| Original | Prompt | Result |
|:--------:|:------:|:------:|
| <img src="https://github.com/user-attachments/assets/187cd903-5d8d-4e3a-97be-0624f104c2c9" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | ➡️ | <img src="https://github.com/user-attachments/assets/3ac87286-7d48-42ac-b7c1-13cfa66408b6" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> |

### Object Detection
| Original | Objects - "photo frame" | Result |
|:--------:|:----------------------:|:------:|
| <img src="https://github.com/user-attachments/assets/43abadd0-8b9f-4c2e-a6c2-49c5ab485ddf" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | ➡️ | <img src="https://github.com/user-attachments/assets/d2f939a5-cdb0-46a7-9b2b-98990f19dae6" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> |

### Wallpaper
| Original | Wallpaper | Result |
|:-------:|:---------:|:------:|
| <img src="https://github.com/user-attachments/assets/64c49cf3-d350-408c-8ec6-8cee1404cf78" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | <img src="https://github.com/user-attachments/assets/41fd54e4-e4a7-4a5d-a4ce-cfc1b9915dc5" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> | <img src="https://github.com/user-attachments/assets/40cdb82d-03cf-482e-924a-724c6c60bdf0" width="300" height="290" style="border-radius:8px; object-fit:cover;"/> |

</div>

<br/>

## 🏗️ Architecture

![Architecture Diagram](docs/images/architecture.png)

<br/>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🌐 Client Layer                                    │
│                    Gradio Web UI  │  REST API Clients                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                         🚀 Flask API Gateway                                 │
│              Authentication  │  Rate Limiting  │  File Handling             │
├─────────────┬─────────────┬─────────────┬──────────────┬────────────────────┤
│  🎨 Wall    │  🖼️ Style   │  ✏️ Inpaint │  🔍 Detect   │  🏷️ Recommend     │
│   Color     │  Transfer   │   Service   │   Service    │    Service         │
├─────────────┴─────────────┴─────────────┴──────────────┴────────────────────┤
│                          🧠 AI Model Layer                                   │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  ┌─────────────┐ │
│   │     SAM     │  │ GroundingDINO│  │ Stable Diffusion  │  │   VGG19     │ │
│   │ (Meta AI)   │  │   (IDEA)     │  │   Inpainting      │  │   ResNet50  │ │
│   └─────────────┘  └──────────────┘  └───────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                       ⚡ PyTorch + CUDA Runtime                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

<br/>

## 💻 Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | GTX 1060 6GB | RTX 3060 12GB | RTX 4080+ 16GB |
| **RAM** | 8 GB | 16 GB | 32 GB |
| **VRAM** | 6 GB | 12 GB | 16+ GB |
| **Storage** | 15 GB | 25 GB | 50 GB (for models) |
| **CPU** | 4 cores | 8 cores | 12+ cores |

> ⚠️ **Note:** CPU-only mode is supported but significantly slower (5-10x). GPU highly recommended.

<br/>

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.10 recommended)
- **CUDA 11.8+** (for GPU acceleration)
- **Git** (for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ai-interior-design.git
cd ai-interior-design
```

### Step 2: Create Virtual Environment

<details>
<summary><b>🪟 Windows</b></summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><b>🐧 Linux / 🍎 macOS</b></summary>

```bash
python -m venv venv
source venv/bin/activate
```

</details>

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Download Model Weights

```bash
python scripts/download_weights.py
```

This downloads:
- **SAM ViT-H checkpoint** (~2.5 GB)
- **GroundingDINO weights** (~700 MB)
- **Stable Diffusion Inpainting** (~4 GB, downloaded on first use)

### Step 5: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings (optional)
# nano .env  # or use your preferred editor
```

### Step 6: Run the Application

#### Option A: Gradio Web UI (Recommended for visual testing)
```bash
python gradio_app.py
```
🌐 Open **http://localhost:7860** in your browser.

#### Option B: Flask API Server (For integration)
```bash
python run.py
```
🔌 API available at **http://localhost:8000**

<br/>

### ✅ Verify Installation

```bash
# Check GPU availability
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Test app initialization
python -c "from app import create_app; app = create_app(); print('✓ App created successfully!')"
```

Expected output:
```
CUDA: True | GPU: NVIDIA GeForce RTX 4060
✓ App created successfully!
```

<br/>

## 📖 API Documentation

### Base URL
```
http://localhost:8000
```

### Health Check
```http
GET /health
```
Returns system status, GPU availability, and version info.

---

### 🎨 Wall Color Change

<details>
<summary><b>POST /api/wall-color/change</b></summary>

Change wall colors in room images using AI-powered segmentation.

**Request:**
```bash
curl -X POST http://localhost:8000/api/wall-color/change \
  -F "image=@room.jpg" \
  -F "color_name=light blue" \
  -F "text_prompt=wall"
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Room image (PNG/JPG) |
| `color_name` | String | Yes | Target color (see `/api/wall-color/colors`) |
| `text_prompt` | String | No | Wall description (default: "wall") |

**Response:** Modified image file (PNG)

</details>

<details>
<summary><b>GET /api/wall-color/colors</b></summary>

Get list of available preset colors.

**Response:**
```json
{
  "colors": ["red", "blue", "light blue", "sage", "beige", "cream", ...]
}
```

</details>

---

### 🖼️ Style Transfer

<details>
<summary><b>POST /api/style-transfer/apply</b></summary>

Apply artistic style transfer to room images.

**Request:**
```bash
curl -X POST http://localhost:8000/api/style-transfer/apply \
  -F "content_image=@room.jpg" \
  -F "style_image=@starry_night.jpg"
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content_image` | File | Yes | Room image |
| `style_image` | File | Yes | Style reference image |

**Response:** Stylized image file (PNG)

</details>

---

### ✏️ Inpainting

<details>
<summary><b>POST /api/inpaint/apply</b></summary>

Replace objects using AI-generated content.

**Request:**
```bash
curl -X POST http://localhost:8000/api/inpaint/apply \
  -F "image=@room.jpg" \
  -F "object_to_detect=sofa" \
  -F "inpaint_prompt=modern minimalist black leather sofa"
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Room image |
| `object_to_detect` | String | Yes | Object to replace |
| `inpaint_prompt` | String | Yes | Description of replacement |

**Response:** Inpainted image file (PNG)

</details>

---

### 🔍 Object Detection

<details>
<summary><b>POST /api/objects/detect</b></summary>

Detect objects in room images.

**Request:**
```bash
curl -X POST http://localhost:8000/api/objects/detect \
  -F "file=@room.jpg"
```

**Response:**
```json
{
  "objects": [
    {"label": "sofa", "confidence": 0.95, "bbox": [100, 200, 400, 350]},
    {"label": "table", "confidence": 0.88, "bbox": [450, 300, 600, 400]}
  ],
  "count": 2
}
```

</details>

---

### 🎭 Wallpaper Application

<details>
<summary><b>POST /api/wallpaper/apply</b></summary>

Apply wallpaper patterns to walls.

**Request:**
```bash
curl -X POST http://localhost:8000/api/wallpaper/apply \
  -F "room_image=@room.jpg" \
  -F "wallpaper_image=@pattern.jpg"
```

**Response:** Room image with wallpaper applied (PNG)

</details>

<br/>

## 📁 Project Structure

```
ai-interior-design/
│
├── 📂 app/                      # Core application
│   ├── __init__.py              # App factory
│   ├── config.py                # Configuration management
│   ├── 📂 api/                  # REST API endpoints
│   │   ├── wall_color.py        # Wall color routes
│   │   ├── style_transfer.py    # Style transfer routes
│   │   ├── object_detection.py  # Detection routes
│   │   ├── inpainting.py        # Inpainting routes
│   │   ├── recommendations.py   # Recommendation routes
│   │   ├── wallpaper.py         # Wallpaper routes
│   │   └── tiles.py             # Tile gallery routes
│   ├── 📂 services/             # Business logic
│   │   ├── segmentation.py      # SAM + GroundingDINO
│   │   ├── style_transfer.py    # Neural style transfer
│   │   ├── inpainting.py        # Stable Diffusion
│   │   └── recommendation.py    # Image similarity
│   ├── 📂 models/               # ML model definitions
│   └── 📂 utils/                # Helper functions
│
├── 📂 data/
│   ├── 📂 samples/              # Sample room images
│   └── 📂 tiles/                # Tile patterns
│
├── 📂 weights/                  # Model weights (gitignored)
├── 📂 media/                    # User uploads (gitignored)
├── 📂 logs/                     # Application logs
├── 📂 docs/                     # Documentation & images
├── 📂 tests/                    # Test suite
├── 📂 scripts/                  # Utility scripts
│
├── gradio_app.py                # 🎨 Gradio web interface
├── run.py                       # 🚀 Flask entry point
├── requirements.txt             # 📦 Dependencies
├── .env.example                 # ⚙️ Environment template
├── .gitignore
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
└── README.md                    # 📖 You are here!
```

<br/>

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|--------------|
| **Backend** | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **AI/ML** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) |
| **Models** | SAM (Meta AI) • GroundingDINO • Stable Diffusion • VGG19 • ResNet50 |
| **UI** | ![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat) |
| **Image Processing** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv) ![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=flat) |
| **GPU** | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) |

</div>

<br/>

## ⚡ Performance Benchmarks

Tested on **NVIDIA RTX 4060** (8GB VRAM):

| Feature | Processing Time | VRAM Usage |
|---------|-----------------|------------|
| Wall Color Change | ~3-5 seconds | ~4 GB |
| Style Transfer (60 epochs) | ~8-12 seconds | ~2 GB |
| Object Detection | ~2-3 seconds | ~3 GB |
| AI Inpainting | ~10-15 seconds | ~6 GB |
| Wallpaper Application | ~4-6 seconds | ~4 GB |

> 💡 **Tip:** First inference may be slower due to model loading. Subsequent runs use cached models.

<br/>

## 🔧 Configuration Options

Create a `.env` file from `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Environment mode |
| `PORT` | `8000` | API server port |
| `USE_GPU` | `true` | Enable GPU acceleration |
| `MODEL_CACHE_DIR` | `./weights` | Model weights directory |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum upload file size |
| `HF_TOKEN` | *(empty)* | Hugging Face token (for gated models) |
| `SD_INPAINTING_MODEL` | `stable-diffusion-v1-5/stable-diffusion-inpainting` | Inpainting model |

<br/>

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html

# Interactive API test script
python scripts/test_api_interactive.py
```

<br/>

## 🐛 Troubleshooting

<details>
<summary><b>❌ CUDA out of memory</b></summary>

**Solutions:**
1. Reduce image resolution before processing
2. Close other GPU-intensive applications
3. Clear CUDA cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
4. Set memory allocation config:
   ```bash
   set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

</details>

<details>
<summary><b>❌ Model download fails</b></summary>

**Solutions:**
1. Check internet connection
2. For gated models, set `HF_TOKEN` in `.env`:
   ```env
   HF_TOKEN=hf_your_token_here
   ```
3. Accept model license on [Hugging Face](https://huggingface.co)
4. Manual download: `python scripts/download_weights.py`

</details>

<details>
<summary><b>❌ GroundingDINO import error</b></summary>

**Solutions:**
```bash
pip uninstall groundingdino-py -y
pip install groundingdino-py

# If issues persist, install from source:
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

</details>

<details>
<summary><b>❌ Gradio not loading</b></summary>

**Solutions:**
1. Check if port 7860 is in use: `netstat -ano | findstr :7860`
2. Try a different port: `python gradio_app.py --server-port 7861`
3. Check firewall settings

</details>

<details>
<summary><b>❌ transformers/diffusers version conflict</b></summary>

**Solution:** Use pinned versions from requirements.txt:
```bash
pip install transformers==4.46.3 diffusers==0.36.0 huggingface-hub==0.36.2
```

</details>

<br/>

## 🗺️ Roadmap

- [x] Wall color visualization with SAM + GroundingDINO
- [x] Neural style transfer (VGG19)
- [x] AI-powered inpainting (Stable Diffusion)
- [x] Object detection (GroundingDINO)
- [x] Wallpaper application
- [x] Gradio web interface
- [x] REST API
- [ ] Floor/ceiling color change
- [ ] 3D room preview
- [ ] Batch image processing
- [ ] Mobile app (React Native)
- [ ] Real-time video processing
- [ ] AR/VR integration
- [ ] Custom model fine-tuning

<br/>

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork** the Project
2. **Create** your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your Changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the Branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

<br/>

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

```
MIT License - Free to use, modify, and distribute with attribution.
```

<br/>

## 🙏 Acknowledgments

This project stands on the shoulders of giants:

| Project | Organization | Purpose |
|---------|--------------|---------|
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) | Meta AI Research | Image segmentation |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | IDEA Research | Zero-shot object detection |
| [Stable Diffusion](https://github.com/CompVis/stable-diffusion) | Stability AI | AI image generation |
| [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Hugging Face | Diffusion model library |
| [Gradio](https://github.com/gradio-app/gradio) | Gradio Team | Web UI framework |
| [Flask](https://flask.palletsprojects.com/) | Pallets Projects | Web framework |

<br/>

## 📬 Contact & Support

<div align="center">

**Have questions? Found a bug? Want to contribute?**

[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-181717?style=for-the-badge&logo=github)](https://github.com/prashantsingh5/ai-interior-design/issues)
[![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-181717?style=for-the-badge&logo=github)](https://github.com/prashantsingh5/ai-interior-design/discussions)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/prashantsingh-ai/)
[![Gmail](https://img.shields.io/badge/Gmail-Contact%20Me-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:prashantsingha96@gmail.com)

</div>

---

<div align="center">

### ⭐ Found this useful? Give it a star!

<br/>

**Built with ❤️ for designers, developers, and AI enthusiasts**

<br/>

[🔝 Back to Top](#-ai-interior-design-studio)

</div>

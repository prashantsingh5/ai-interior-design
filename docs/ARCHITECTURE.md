# AI Interior Design Studio Architecture

This document describes the production architecture of the project based on the current implementation in:
- `run.py`
- `gradio_app.py`
- `app/__init__.py`
- `app/api/*.py`
- `app/services/*.py`
- `app/utils/*.py`

## 1. System Context

```mermaid
flowchart TB
    U1[Web User] --> G[Gradio UI\ngradio_app.py]
    U2[API Consumer] --> F[Flask API\nrun.py + app factory]

    G --> F

    F --> A1[Wall Color API]
    F --> A2[Object Detection API]
    F --> A3[Wallpaper API]
    F --> A4[Inpainting API]
    F --> A5[Style Transfer API]
    F --> A6[Recommendations API]
    F --> A7[Tiles API]

    A1 --> S1[SegmentationService]
    A2 --> S1
    A3 --> S1
    A4 --> S2[InpaintingService]
    A5 --> S3[StyleTransferService]
    A6 --> S4[RecommendationService]
    A7 --> D2[Tile Assets\ndata/tiles]

    S2 --> S1

    S1 --> M1[GroundingDINO]
    S1 --> M2[SAM]
    S2 --> M3[Stable Diffusion Inpainting]
    S3 --> M4[VGG19]
    S4 --> M5[ResNet50 + sklearn]

    M1 --> W[Weights + HF Hub]
    M2 --> W
    M3 --> W

    F --> D1[Media Storage\nmedia/*]
    S4 --> D3[Image Features\ndata/image_features.csv]
```

## 2. Runtime Layers

1. Interface Layer
- `gradio_app.py` provides an interactive UI with lazy service loading.
- Flask endpoints provide REST interfaces for all capabilities.

2. API Layer (Flask Blueprints)
- `app/api/wall_color.py`
- `app/api/object_detection.py`
- `app/api/wallpaper.py`
- `app/api/inpainting.py`
- `app/api/style_transfer.py`
- `app/api/recommendations.py`
- `app/api/tiles.py`

3. Service Layer
- `SegmentationService`: wall/object detection, wall mask generation, wallpaper overlay.
- `InpaintingService`: object mask + diffusion inpainting.
- `StyleTransferService`: neural style transfer optimization loop.
- `RecommendationService`: embeddings, similarity search, image analysis.

4. Model Layer
- GroundingDINO for text-guided detection.
- SAM for segmentation masks.
- Stable Diffusion Inpainting for object replacement.
- VGG19 for style transfer.
- ResNet50 (TensorFlow) for recommendation embeddings.

5. Data/Storage Layer
- `media/` for request inputs and generated outputs.
- `weights/` and `MODEL_CACHE_DIR` for model artifacts.
- `data/tiles/` for wallpaper/tile gallery assets.
- `data/image_features.csv` for recommendation lookup.

## 3. Endpoint to Service Mapping

| Endpoint | Blueprint File | Service | Output |
|---|---|---|---|
| `POST /api/wall-color/change` | `app/api/wall_color.py` | `SegmentationService.change_wall_color` | PNG image |
| `GET /api/wall-color/colors` | `app/api/wall_color.py` | `Config.COLOR_MAP` | JSON |
| `POST /api/objects/detect` | `app/api/object_detection.py` | `SegmentationService.detect_objects` | JSON |
| `GET /api/objects/default-objects` | `app/api/object_detection.py` | `Config.DEFAULT_DETECTABLE_OBJECTS` | JSON |
| `POST /api/wallpaper/apply` | `app/api/wallpaper.py` | `SegmentationService.apply_wallpaper` | JPG image |
| `POST /api/inpaint/apply` | `app/api/inpainting.py` | `InpaintingService.inpaint` | PNG image |
| `POST /api/style-transfer/apply` | `app/api/style_transfer.py` | `StyleTransferService.transfer_style` | JPG image |
| `POST /api/recommendations/similar` | `app/api/recommendations.py` | `RecommendationService.get_recommendations` | JSON |
| `POST /api/recommendations/analyze` | `app/api/recommendations.py` | `RecommendationService.analyze_image` | JSON |
| `GET /api/recommendations/categories` | `app/api/recommendations.py` | Static categories | JSON |
| `GET /api/tiles/gallery` | `app/api/tiles.py` | Filesystem tile scan + base64 thumbnails | JSON |
| `GET /api/tiles/image/<filename>` | `app/api/tiles.py` | Filesystem direct image | Image |

## 4. Key Processing Flows

### 4.1 Wall Color Change
```mermaid
sequenceDiagram
    participant C as Client
    participant API as Wall Color API
    participant SEG as SegmentationService
    participant GD as GroundingDINO
    participant SAM as SAM

    C->>API: POST image + text_prompt + color
    API->>SEG: change_wall_color(...)
    SEG->>GD: detect wall boxes from text
    SEG->>SAM: generate mask from boxes
    SEG->>SEG: blend target color on mask
    SEG-->>API: output image path
    API-->>C: PNG result
```

### 4.2 Object Replacement Inpainting
```mermaid
sequenceDiagram
    participant C as Client
    participant API as Inpainting API
    participant INP as InpaintingService
    participant SEG as SegmentationService
    participant SD as Stable Diffusion

    C->>API: POST image + object_to_replace + prompt
    API->>INP: inpaint(...)
    INP->>SEG: detect object + SAM mask
    INP->>SD: run inpainting with prompt + mask
    INP-->>API: generated image path
    API-->>C: PNG result
```

### 4.3 Recommendations
```mermaid
sequenceDiagram
    participant C as Client
    participant API as Recommendations API
    participant REC as RecommendationService
    participant CSV as image_features.csv

    C->>API: POST image + category + limit
    API->>REC: get_recommendations(...)
    REC->>REC: extract embedding + color + contrast
    REC->>CSV: load/filter precomputed features
    REC->>REC: cosine similarity ranking
    REC-->>API: top-N recommendations + analysis
    API-->>C: JSON payload
```

## 5. Deployment View (Current)

```mermaid
flowchart LR
    subgraph Host[Single Python Host]
        API[Flask App Process]
        UI[Gradio Process]
        FS[(Local Filesystem)]
        GPU[(CUDA GPU Optional)]
    end

    API <--> FS
    UI <--> FS
    API <--> GPU
    UI <--> GPU
```

Notes:
- Flask and Gradio can run together in development; production should use a reverse proxy and separate process management.
- Model loading is mostly lazy/singleton in service classes, which reduces startup cost but increases first-request latency.

## 6. Architectural Strengths

- Clean separation between API endpoints and AI service logic.
- Reusable segmentation core across wall color, detection, wallpaper, and inpainting mask generation.
- Lazy model loading avoids loading all heavy models at boot.
- Utility layer centralizes file I/O and output path conventions.

## 7. Improvement Roadmap

1. Add an async job queue (Celery/RQ + Redis) for long-running inference tasks.
2. Move generated media to object storage (S3/Azure Blob) with signed URLs.
3. Add model warm-up endpoint and health checks per model.
4. Split TensorFlow recommendation service into optional microservice to reduce base memory footprint.
5. Add request tracing and per-endpoint latency metrics.
6. Add API auth/rate limiting (the README diagram mentions this, but it is not currently implemented in code).

## 8. GitHub Integration

Recommended repo links:
- In `README.md`, link this file under Architecture: `docs/ARCHITECTURE.md`.
- Keep Mermaid diagrams in Markdown so they render natively in GitHub without external images.

# DeepfakeDetector CLIP Edition

A deepfake detection system using OpenAI's CLIP (Contrastive Language-Image Pre-Training) as a feature extractor with a custom classifier. This project leverages CLIP's powerful visual understanding for detecting manipulated images and videos.

## Features

* **CLIP Architecture**: Uses OpenAI's CLIP (ViT-B/32) for robust frame encoding
* **Images & Videos**: Works with both image files and video files
* **Face-Focused**: Automatically detects and crops faces using MTCNN
* **Testing Metrics**: Comprehensive evaluation with accuracy, precision, recall, F1-score, and confusion matrix
* **Interactive App**: Includes a Streamlit web interface for easy testing
* **Pre-trained Ready**: Uses CLIP's pre-trained weights without requiring training

## Installation

1. Clone the repository:

```bash
git clone https://github.com/HamzaKhan760/AIGenDetector.git
cd AIGenDetector
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Run the Web App

The easiest way to use the detector is via the Streamlit app.

```bash
streamlit run apps/streamlit_app.py
```

Upload an image or video file (`.jpg`, `.png`, `.mp4`, `.mov`, `.avi`) to analyze it.

### 2. Test and Get Metrics

To evaluate the model on your dataset and get accuracy metrics:

1. **Prepare Test Data**: Organize your images and/or videos:
   ```
   data/test/real/  (put real images/videos here)
   data/test/fake/  (put fake images/videos here)
   ```

   Supported formats:
   - Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
   - Videos: `.mp4`, `.mov`, `.avi`, `.mkv`

2. **Run Evaluation**:

```bash
python scripts/evaluate.py
```

This will output:
- Overall accuracy percentage
- Precision, Recall, F1-Score
- Confusion Matrix
- Per-file predictions saved to `results/predictions.csv`
- Detailed metrics report saved to `results/metrics_report.txt`

### 3. Process Single Image or Video

To test a single file from command line:

```bash
python scripts/predict_video.py --file_path path/to/your/file.jpg
# or
python scripts/predict_video.py --file_path path/to/your/video.mp4
```

## Project Structure

```
DeepfakeDetector_CLIP/
├── models/
│   └── clip_detector.py       # CLIP-based detector model
├── utils/
│   ├── preprocessing.py       # Video processing and face detection
│   ├── face_detector.py       # MTCNN face detection wrapper
│   └── metrics.py             # Evaluation metrics
├── apps/
│   └── streamlit_app.py       # Web interface
├── scripts/
│   ├── evaluate.py            # Batch evaluation script
│   └── predict_video.py       # Single video prediction
├── data/
│   └── test/
│       ├── real/              # Real videos for testing
│       └── fake/              # Fake videos for testing
├── results/                    # Evaluation results
├── requirements.txt
└── README.md
```

## How It Works

1. **Frame Extraction**: 
   - Images: Processed directly as single frames
   - Videos: Processed frame-by-frame (samples multiple frames)
2. **Face Detection**: MTCNN detects and crops faces from each frame
3. **CLIP Encoding**: Cropped faces are encoded using CLIP's vision transformer
4. **Classification**: A zero-shot classifier uses CLIP features with text prompts to predict real/fake
5. **Aggregation**: For videos, frame-level predictions are averaged for final prediction

## Testing Metrics Explained

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Of all videos predicted as fake, what percentage were actually fake?
- **Recall**: Of all actual fake videos, what percentage did we detect?
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows true positives, false positives, true negatives, false negatives

## Requirements

Key dependencies:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `transformers>=4.30.0`
- `opencv-python>=4.8.0`
- `facenet-pytorch>=2.5.0` (for MTCNN)
- `streamlit>=1.25.0`
- `pillow>=9.5.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `tqdm>=4.65.0`

## Notes

- CLIP is used in a zero-shot mode with text prompts - no training required
- For images, the model analyzes the single image
- For videos, the model analyzes multiple frames for robust predictions
- Face detection ensures the model focuses on relevant facial regions
- Results are aggregated across frames for final predictions

## License

MIT License
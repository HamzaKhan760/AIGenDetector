# DeepfakeDetector CLIP Edition

A deepfake detection system using OpenAI's CLIP (Contrastive Language-Image Pre-Training) as a feature extractor with a custom classifier. This project leverages CLIP's powerful visual understanding for detecting manipulated videos.

## Features

* **CLIP Architecture**: Uses OpenAI's CLIP (ViT-B/32) for robust frame encoding
* **Face-Focused**: Automatically detects and crops faces using MTCNN
* **Testing Metrics**: Comprehensive evaluation with accuracy, precision, recall, F1-score, and confusion matrix
* **Interactive App**: Includes a Streamlit web interface for easy testing
* **Pre-trained Ready**: Uses CLIP's pre-trained weights without requiring training

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DeepfakeDetector_CLIP.git
cd DeepfakeDetector_CLIP
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
pip install -r requirements.txt
```

## Usage

### 1. Run the Web App

The easiest way to use the detector is via the Streamlit app.

```bash
python -m streamlit run apps/streamlit_app.py
```

Upload a video file (`.mp4`, `.mov`, `.avi`) to analyze it.

### 2. Test and Get Metrics

To evaluate the model on your dataset and get accuracy metrics:

1. **Prepare Test Data**: Organize your videos:
   ```
   data/test/real/  (put real videos here)
   data/test/fake/  (put fake videos here)
   ```

2. **Run Evaluation**:

```bash
python scripts/evaluate.py
```

This will output:
- Overall accuracy percentage
- Precision, Recall, F1-Score
- Confusion Matrix
- Per-video predictions saved to `results/predictions.csv`
- Detailed metrics report saved to `results/metrics_report.txt`

### 3. Process Single Video

To test a single video from command line:

```bash
python scripts/predict_video.py --video_path path/to/your/video.mp4
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

1. **Frame Extraction**: Videos are processed frame-by-frame
2. **Face Detection**: MTCNN detects and crops faces from each frame
3. **CLIP Encoding**: Cropped faces are encoded using CLIP's vision transformer
4. **Classification**: A simple classifier uses CLIP features to predict real/fake
5. **Aggregation**: Frame-level predictions are averaged for final video prediction

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

- CLIP is used in a zero-shot or feature extraction mode - no training required
- The model analyzes multiple frames per video for robust predictions
- Face detection ensures the model focuses on relevant facial regions
- Results are aggregated across frames for final video-level prediction

## License

MIT License
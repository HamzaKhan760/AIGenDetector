"""
Script to predict if a single image or video is fake or real
"""

import argparse
import sys
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from models.clip_detector import SimpleZeroShotCLIPDetector
from utils.preprocessing import extract_frames_from_video, preprocess_frame
from utils.face_detector import FaceDetector


def predict_file(file_path, max_frames=10, threshold=0.5):
    """
    Predict if an image or video is fake or real
    
    Args:
        file_path: Path to image or video file
        max_frames: Maximum frames to sample (for videos)
        threshold: Classification threshold
    
    Returns:
        Tuple of (is_fake, confidence_score, frame_scores)
    """
    print(f"Analyzing file: {file_path}")
    print("="*50)
    
    # Determine if image or video
    file_ext = Path(file_path).suffix.lower()
    is_image = file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    # Initialize models
    print("Loading CLIP model...")
    detector = SimpleZeroShotCLIPDetector()
    face_detector = FaceDetector()
    
    # Extract frames
    if is_image:
        print("Processing image...")
        img = cv2.imread(file_path)
        if img is None:
            print("ERROR: Could not load image")
            return None, 0.0, []
        frames = [img]
        print("Image loaded successfully")
    else:
        print(f"Extracting up to {max_frames} frames from video...")
        frames = extract_frames_from_video(file_path, max_frames=max_frames)
        
        if not frames:
            print("ERROR: Could not extract frames from video")
            return None, 0.0, []
        
        print(f"Extracted {len(frames)} frames")
    
    # Detect and crop faces
    print("Detecting faces...")
    face_frames = []
    
    for i, frame in enumerate(frames):
        face = face_detector.detect_and_crop_face(frame)
        if face is not None:
            face_pil = preprocess_frame(face)
            face_frames.append(face_pil)
            print(f"  Frame {i+1}: Face detected ✓")
        else:
            print(f"  Frame {i+1}: No face detected ✗")
    
    if not face_frames:
        print("ERROR: No faces detected in any frame")
        return None, 0.0, []
    
    print(f"\nTotal faces detected: {len(face_frames)}/{len(frames)}")
    
    # Predict
    print("\nAnalyzing frames with CLIP...")
    avg_score, frame_scores = detector.predict_frames(face_frames)
    
    is_fake = avg_score > threshold
    
    # Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Average Fake Score: {avg_score:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Verdict: {'FAKE' if is_fake else 'REAL'}")
    print(f"Confidence: {(avg_score if is_fake else 1-avg_score)*100:.2f}%")
    
    print("\nFrame-by-frame scores:")
    for i, score in enumerate(frame_scores):
        verdict = "FAKE" if score > threshold else "REAL"
        print(f"  Frame {i+1}: {score:.4f} ({verdict})")
    
    print("\nStatistics:")
    print(f"  Min score: {min(frame_scores):.4f}")
    print(f"  Max score: {max(frame_scores):.4f}")
    print(f"  Std dev: {sum((s - avg_score)**2 for s in frame_scores) / len(frame_scores)**0.5:.4f}")
    
    return is_fake, avg_score, frame_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an image or video is fake or real")
    parser.add_argument("--file_path", type=str, required=True,
                       help="Path to image or video file")
    parser.add_argument("--max_frames", type=int, default=10,
                       help="Maximum frames to analyze (for videos)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file_path).exists():
        print(f"ERROR: File not found: {args.file_path}")
        sys.exit(1)
    
    # Run prediction
    is_fake, score, frame_scores = predict_file(
        args.file_path,
        max_frames=args.max_frames,
        threshold=args.threshold
    )
    
    # Exit code based on result
    sys.exit(0 if is_fake is not None else 1)
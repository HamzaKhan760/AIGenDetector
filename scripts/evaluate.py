"""
Evaluation script for CLIP Deepfake Detector
Generates accuracy metrics on test dataset (supports both images and videos)
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import cv2
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.clip_detector import SimpleZeroShotCLIPDetector
from utils.preprocessing import extract_frames_from_video, preprocess_frame
from utils.face_detector import FaceDetector


def process_image(image_path):
    """
    Process a single image file
    Returns list containing single frame
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    return [img]


def evaluate_model(
    test_real_dir="data/test/real",
    test_fake_dir="data/test/fake",
    max_frames=10,
    threshold=0.5,
    output_dir="results"
):
    """
    Evaluate the CLIP detector on test dataset
    
    Args:
        test_real_dir: Directory containing real videos
        test_fake_dir: Directory containing fake videos
        max_frames: Maximum frames to sample per video
        threshold: Classification threshold (>threshold = fake)
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and face detector
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    detector = SimpleZeroShotCLIPDetector()
    face_detector = FaceDetector()
    
    # Collect all video and image paths
    video_exts = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    
    real_files = []
    fake_files = []
    
    for ext in video_exts + image_exts:
        real_files.extend(list(Path(test_real_dir).glob(ext)))
        fake_files.extend(list(Path(test_fake_dir).glob(ext)))
    
    print(f"\nFound {len(real_files)} real files and {len(fake_files)} fake files")
    
    if len(real_files) == 0 or len(fake_files) == 0:
        print("Error: No files found in test directories!")
        print(f"Real files dir: {test_real_dir}")
        print(f"Fake files dir: {test_fake_dir}")
        return
    
    # Store predictions
    results = []
    y_true = []
    y_pred = []
    y_scores = []
    
    # Process real files (label = 0)
    print("\n" + "="*50)
    print("Processing REAL files...")
    print("="*50)
    for file_path in tqdm(real_files):
        try:
            # Check if it's an image or video
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                # Process as image
                frames = process_image(file_path)
            else:
                # Process as video
                frames = extract_frames_from_video(str(file_path), max_frames=max_frames)
            
            if not frames:
                print(f"No frames extracted from {file_path.name}")
                continue
            
            # Detect faces and preprocess
            face_frames = []
            for frame in frames:
                face = face_detector.detect_and_crop_face(frame)
                if face is not None:
                    face_pil = preprocess_frame(face)
                    face_frames.append(face_pil)
            
            if not face_frames:
                print(f"No faces detected in {file_path.name}")
                continue
            
            # Predict
            avg_score, frame_scores = detector.predict_frames(face_frames)
            prediction = 1 if avg_score > threshold else 0
            
            y_true.append(0)  # Real
            y_pred.append(prediction)
            y_scores.append(avg_score)
            
            results.append({
                'file': file_path.name,
                'type': 'image' if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] else 'video',
                'true_label': 'real',
                'predicted_label': 'fake' if prediction == 1 else 'real',
                'confidence_score': avg_score,
                'num_frames': len(face_frames),
                'correct': prediction == 0
            })
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    # Process fake files (label = 1)
    print("\n" + "="*50)
    print("Processing FAKE files...")
    print("="*50)
    for file_path in tqdm(fake_files):
        try:
            # Check if it's an image or video
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                # Process as image
                frames = process_image(file_path)
            else:
                # Process as video
                frames = extract_frames_from_video(str(file_path), max_frames=max_frames)
            
            if not frames:
                print(f"No frames extracted from {file_path.name}")
                continue
            
            # Detect faces and preprocess
            face_frames = []
            for frame in frames:
                face = face_detector.detect_and_crop_face(frame)
                if face is not None:
                    face_pil = preprocess_frame(face)
                    face_frames.append(face_pil)
            
            if not face_frames:
                print(f"No faces detected in {file_path.name}")
                continue
            
            # Predict
            avg_score, frame_scores = detector.predict_frames(face_frames)
            prediction = 1 if avg_score > threshold else 0
            
            y_true.append(1)  # Fake
            y_pred.append(prediction)
            y_scores.append(avg_score)
            
            results.append({
                'file': file_path.name,
                'type': 'image' if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] else 'video',
                'true_label': 'fake',
                'predicted_label': 'fake' if prediction == 1 else 'real',
                'confidence_score': avg_score,
                'num_frames': len(face_frames),
                'correct': prediction == 1
            })
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if len(y_true) == 0:
        print("No files were successfully processed!")
        return
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Real    Fake")
    print(f"Actual Real   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"       Fake   {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Real', 'Fake'],
                                zero_division=0))
    
    # Save results
    df = pd.DataFrame(results)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")
    
    # Save metrics report
    report_path = os.path.join(output_dir, 'metrics_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("CLIP DEEPFAKE DETECTOR - EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total files processed: {len(y_true)}\n")
        f.write(f"Real files: {len([y for y in y_true if y == 0])}\n")
        f.write(f"Fake files: {len([y for y in y_true if y == 1])}\n\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("                Predicted\n")
        f.write("              Real    Fake\n")
        f.write(f"Actual Real   {cm[0][0]:4d}    {cm[0][1]:4d}\n")
        f.write(f"       Fake   {cm[1][0]:4d}    {cm[1][1]:4d}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(classification_report(y_true, y_pred, 
                                     target_names=['Real', 'Fake'],
                                     zero_division=0))
    
    print(f"Metrics report saved to: {report_path}")
    
    # Print some example predictions
    print("\n" + "="*50)
    print("Sample Predictions:")
    print("="*50)
    print(df.head(10).to_string(index=False))
    
    return accuracy, precision, recall, f1, cm, df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CLIP Deepfake Detector")
    parser.add_argument("--real_dir", type=str, default="data/test/real",
                       help="Directory containing real videos")
    parser.add_argument("--fake_dir", type=str, default="data/test/fake",
                       help="Directory containing fake videos")
    parser.add_argument("--max_frames", type=int, default=10,
                       help="Maximum frames to sample per video")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluate_model(
        test_real_dir=args.real_dir,
        test_fake_dir=args.fake_dir,
        max_frames=args.max_frames,
        threshold=args.threshold,
        output_dir=args.output_dir
    )
"""
Preprocessing utilities for video and frame processing
"""

import cv2
import numpy as np
from PIL import Image


def extract_frames_from_video(video_path, max_frames=10, sample_method='uniform'):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        sample_method: 'uniform' or 'random'
    
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Error: Video has no frames {video_path}")
        return []
    
    # Determine which frames to extract
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        if sample_method == 'uniform':
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:  # random
            frame_indices = sorted(np.random.choice(total_frames, max_frames, replace=False))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess frame for CLIP model
    Converts BGR numpy array to RGB PIL Image
    
    Args:
        frame: Frame as numpy array (BGR format from OpenCV)
        target_size: Target size for resizing
    
    Returns:
        PIL Image in RGB format
    """
    # Convert BGR to RGB
    if isinstance(frame, np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
    else:
        pil_image = frame
    
    # Resize
    pil_image = pil_image.resize(target_size, Image.BILINEAR)
    
    return pil_image


def apply_augmentation(frame, augmentation_type='none'):
    """
    Apply augmentation to frame
    
    Args:
        frame: Frame as numpy array
        augmentation_type: 'none', 'blur', 'compression', 'noise'
    
    Returns:
        Augmented frame
    """
    if augmentation_type == 'blur':
        return cv2.GaussianBlur(frame, (5, 5), 0)
    elif augmentation_type == 'compression':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encimg, 1)
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)
    else:
        return frame


def frames_to_video(frames, output_path, fps=30):
    """
    Save frames as video
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Output video path
        fps: Frames per second
    """
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
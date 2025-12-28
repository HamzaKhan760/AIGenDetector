"""
Face detection utilities using MTCNN
"""

import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
from PIL import Image


class FaceDetector:
    """
    Face detector wrapper using MTCNN
    """
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.mtcnn = MTCNN(
            select_largest=True,
            device=device,
            post_process=False
        )
    
    def detect_face_box(self, frame):
        """
        Detect face bounding box in frame
        
        Args:
            frame: Frame as numpy array (BGR or RGB)
        
        Returns:
            Bounding box as [x1, y1, x2, y2] or None
        """
        # Convert to RGB if BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume BGR from OpenCV
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Detect face
        boxes, _ = self.mtcnn.detect(pil_image)
        
        if boxes is not None and len(boxes) > 0:
            # Return first (largest) face box
            box = boxes[0]
            return box.astype(int)
        
        return None
    
    def detect_and_crop_face(self, frame, margin=0.2):
        """
        Detect and crop face from frame with margin
        
        Args:
            frame: Frame as numpy array (BGR format)
            margin: Margin to add around face (as fraction of face size)
        
        Returns:
            Cropped face as numpy array or None
        """
        box = self.detect_face_box(frame)
        
        if box is None:
            return None
        
        x1, y1, x2, y2 = box
        
        # Add margin
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(frame.shape[1], x2 + margin_x)
        y2 = min(frame.shape[0], y2 + margin_y)
        
        # Crop face
        face = frame[y1:y2, x1:x2]
        
        return face
    
    def detect_multiple_faces(self, frame):
        """
        Detect all faces in frame
        
        Args:
            frame: Frame as numpy array
        
        Returns:
            List of bounding boxes
        """
        # Convert to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        pil_image = Image.fromarray(frame_rgb)
        
        boxes, _ = self.mtcnn.detect(pil_image)
        
        if boxes is not None:
            return boxes.astype(int).tolist()
        
        return []
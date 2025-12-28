"""
CLIP-based Deepfake Detector Model
Uses OpenAI's CLIP for feature extraction
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


class CLIPDetector(nn.Module):
    """
    Deepfake detector using CLIP vision encoder
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_clip=True):
        super().__init__()
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP weights if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get CLIP vision embedding dimension
        self.embedding_dim = self.clip_model.config.projection_dim
        
        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode_image(self, image):
        """
        Encode a single PIL Image using CLIP
        Returns normalized embedding vector
        """
        # Process image for CLIP
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to same device as model
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # Get CLIP vision features
        with torch.no_grad():
            vision_outputs = self.clip_model.get_image_features(**inputs)
        
        # Normalize features
        vision_outputs = vision_outputs / vision_outputs.norm(dim=-1, keepdim=True)
        
        return vision_outputs
    
    def forward(self, images):
        """
        Forward pass for batch of images
        images: list of PIL Images or tensor
        """
        if isinstance(images, list):
            # Batch encode images
            embeddings = []
            for img in images:
                emb = self.encode_image(img)
                embeddings.append(emb)
            embeddings = torch.cat(embeddings, dim=0)
        else:
            # Already processed tensor
            embeddings = images
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits
    
    def predict_frame(self, frame_pil):
        """
        Predict if a single frame (PIL Image) is fake or real
        Returns probability of being fake (0-1)
        """
        self.eval()
        with torch.no_grad():
            embedding = self.encode_image(frame_pil)
            prob = self.classifier(embedding)
        return prob.item()
    
    def predict_frames(self, frames_pil_list):
        """
        Predict for multiple frames and aggregate
        Returns average probability of being fake
        """
        self.eval()
        probs = []
        
        with torch.no_grad():
            for frame in frames_pil_list:
                embedding = self.encode_image(frame)
                prob = self.classifier(embedding)
                probs.append(prob.item())
        
        return np.mean(probs), probs


class SimpleZeroShotCLIPDetector:
    """
    Zero-shot CLIP detector using text prompts (no training needed)
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        
        # Text prompts for classification
        self.real_prompts = [
            "a photo of a real human face",
            "an authentic person",
            "a genuine photograph of a person"
        ]
        self.fake_prompts = [
            "a photo of a deepfake face",
            "an AI-generated face",
            "a synthetic person",
            "a computer-generated face"
        ]
    
    def predict_frame(self, frame_pil):
        """
        Zero-shot prediction using text-image similarity
        """
        # Prepare inputs
        inputs = self.processor(
            text=self.real_prompts + self.fake_prompts,
            images=frame_pil,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get CLIP outputs
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Average probabilities for real vs fake
        real_prob = probs[0, :len(self.real_prompts)].mean().item()
        fake_prob = probs[0, len(self.real_prompts):].mean().item()
        
        # Return probability of being fake
        return fake_prob / (real_prob + fake_prob)
    
    def predict_frames(self, frames_pil_list):
        """
        Predict for multiple frames
        """
        probs = [self.predict_frame(frame) for frame in frames_pil_list]
        return np.mean(probs), probs
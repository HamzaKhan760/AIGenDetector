"""
Streamlit web app for CLIP Deepfake Detector
Supports both images and videos
"""

import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.clip_detector import SimpleZeroShotCLIPDetector
from utils.preprocessing import extract_frames_from_video, preprocess_frame
from utils.face_detector import FaceDetector


# Page config
st.set_page_config(
    page_title="CLIP Deepfake Detector",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 CLIP Deepfake Detector")
st.markdown("Upload an image or video to detect if it contains deepfake manipulation")

# Sidebar
st.sidebar.header("Settings")
max_frames = st.sidebar.slider("Max frames to analyze", 5, 30, 10)
threshold = st.sidebar.slider("Detection threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This detector uses OpenAI's CLIP model to analyze images and video frames "
    "and detect potential deepfake manipulation. It focuses on facial regions "
    "and aggregates predictions across multiple frames for videos."
)


# Initialize models (cache to avoid reloading)
@st.cache_resource
def load_models():
    detector = SimpleZeroShotCLIPDetector()
    face_detector = FaceDetector()
    return detector, face_detector


# Main app
def main():
    # Load models
    with st.spinner("Loading models..."):
        detector, face_detector = load_models()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['mp4', 'mov', 'avi', 'jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Upload an image or video file to analyze"
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_ext = uploaded_file.name.split('.')[-1].lower()
        is_image = file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        
        # Save uploaded file temporarily
        suffix = f'.{file_ext}'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Display file
            if is_image:
                st.image(uploaded_file, use_container_width=True)
            else:
                st.video(uploaded_file)
            
            # Analyze button
            if st.button("🔍 Analyze", type="primary"):
                analyze_file(tmp_path, detector, face_detector, max_frames, threshold, is_image)
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def analyze_file(file_path, detector, face_detector, max_frames, threshold, is_image):
    """Analyze image or video and display results"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Extract frames
    if is_image:
        status_text.text("Loading image...")
        progress_bar.progress(10)
        
        img = cv2.imread(file_path)
        if img is None:
            st.error("❌ Could not load image")
            return
        frames = [img]
        status_text.text("Image loaded")
    else:
        status_text.text("Extracting frames...")
        progress_bar.progress(10)
        
        frames = extract_frames_from_video(file_path, max_frames=max_frames)
        
        if not frames:
            st.error("❌ Could not extract frames from video")
            return
        
        status_text.text(f"Extracted {len(frames)} frames")
    
    progress_bar.progress(30)
    
    # Detect faces
    status_text.text("Detecting faces...")
    face_frames = []
    
    for i, frame in enumerate(frames):
        face = face_detector.detect_and_crop_face(frame)
        if face is not None:
            face_pil = preprocess_frame(face)
            face_frames.append(face_pil)
        
        progress_bar.progress(30 + int(40 * (i + 1) / len(frames)))
    
    if not face_frames:
        st.warning("⚠️ No faces detected in the file")
        return
    
    status_text.text(f"Detected faces in {len(face_frames)} frames")
    
    # Predict
    status_text.text("Analyzing with CLIP...")
    progress_bar.progress(70)
    
    avg_score, frame_scores = detector.predict_frames(face_frames)
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Results")
    
    # Determine verdict
    is_fake = avg_score > threshold
    
    if is_fake:
        st.error(f"⚠️ **FAKE DETECTED** (Confidence: {avg_score:.1%})")
        st.markdown(
            f"The video shows signs of manipulation with a confidence score of **{avg_score:.1%}**"
        )
    else:
        st.success(f"✅ **APPEARS REAL** (Confidence: {(1-avg_score):.1%})")
        st.markdown(
            f"The video appears authentic with a confidence score of **{(1-avg_score):.1%}**"
        )
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frames/Images Analyzed", len(face_frames))
    with col2:
        st.metric("Fake Score", f"{avg_score:.1%}")
    with col3:
        st.metric("Verdict", "FAKE" if is_fake else "REAL")
    
    # Frame-by-frame scores (only show if multiple frames)
    if len(frame_scores) > 1:
        with st.expander("📈 Frame-by-frame Analysis"):
            st.line_chart(frame_scores)
            st.caption(f"Average score: {avg_score:.3f} | Threshold: {threshold:.3f}")
            
            # Statistics
            st.markdown("**Statistics:**")
            st.write(f"- Min score: {min(frame_scores):.3f}")
            st.write(f"- Max score: {max(frame_scores):.3f}")
            st.write(f"- Std dev: {np.std(frame_scores):.3f}")
    
    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This is an experimental tool and should not be used "
        "as the sole basis for determining image/video authenticity. Results may vary "
        "based on quality and other factors."
    )


if __name__ == "__main__":
    main()
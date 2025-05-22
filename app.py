import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

# Constants
IMG_SIZE = 64
FRAMES = 30

# Load saved model
model = load_model("cash_lifting_detector.h5")

# Helper: Extract 30 evenly spaced frames
def extract_frames(video_path, target_frames=FRAMES, size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames >= target_frames:
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        pad_needed = target_frames - total_frames

    frames = []
    idx = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in indices:
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            idx += 1
        count += 1

    cap.release()

    while len(frames) < target_frames:
        frames.append(np.zeros((size, size, 3), dtype=np.uint8))

    return np.array(frames)

# Streamlit UI
st.set_page_config(page_title="Cash Lifting Detector", layout="centered")
st.title("💸 Cash Lifting Detection App")

uploaded_file = st.file_uploader("Upload a video (.mp4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Display the video
    st.video(uploaded_file)

    if st.button("Analyze Video"):
        with st.spinner("Analyzing..."):
            # Extract and preprocess
            frames = extract_frames(video_path)
            frames = np.expand_dims(frames, axis=0)
            frames = frames / 255.0

            # Predict
            prediction = model.predict(frames)
            class_idx = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100

            label = "Abnormal (Cash Lifting Detected)" if class_idx == 1 else "Normal"
            st.success(f"Prediction: **{label}** with {confidence:.2f}% confidence")
else:
    st.info("Please upload a video to begin.")

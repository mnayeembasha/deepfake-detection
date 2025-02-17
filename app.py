from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import torch
import torchaudio
import librosa
import numpy as np
import cv2
import torchvision.transforms as transforms
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from torchvision import models
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

### -------------------- AUDIO DEEPFAKE DETECTION SETUP -------------------- ###

# Load Wav2Vec2 Model for Audio DeepFake Detection
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model_audio = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_audio.eval()  # set to evaluation mode

# Audio Preprocessing Function
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    return input_values

# Detect Fake Audio
def detect_fake_voice(audio_path):
    input_values = preprocess_audio(audio_path)
    with torch.no_grad():
        logits = model_audio(input_values).logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "FAKE VOICE" if prediction == 1 else "REAL VOICE"

### -------------------- VIDEO DEEPFAKE DETECTION SETUP -------------------- ###

# Load EfficientNet Model for Video DeepFake Detection
model_video = models.efficientnet_b0(pretrained=True)
model_video.eval()

# Define transform for video frame processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Detect Fake Video from a single frame
def detect_fake_video(frame):
    # Convert OpenCV BGR image to PIL RGB image
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = transform(frame).unsqueeze(0)
    with torch.no_grad():
        output = model_video(frame)
    prediction = torch.argmax(output, dim=-1).item()
    return "FAKE VIDEO" if prediction == 1 else "REAL VIDEO"

### -------------------- FLASK ENDPOINTS -------------------- ###

# Home route to render index.html
@app.route("/")
def home():
    return render_template("index.html")

# Endpoint for audio file upload and detection
@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No audio file selected"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    result = detect_fake_voice(file_path)
    return jsonify({"result": result})

# Endpoint for video file upload and detection
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video file selected"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    # Open video and extract the first frame
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Could not read video file"}), 400
    result = detect_fake_video(frame)
    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)

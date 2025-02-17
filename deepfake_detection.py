from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Wav2Vec2 model
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model_audio = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Audio Preprocessing Function
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    return input_values

# Detect Fake Voice
def detect_fake_voice(audio_path):
    input_values = preprocess_audio(audio_path)
    with torch.no_grad():
        logits = model_audio(input_values).logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "FAKE VOICE" if prediction == 1 else "REAL VOICE"

# API Endpoint for Audio Upload
@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    result = detect_fake_voice(file_path)
    return jsonify({"result": result})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

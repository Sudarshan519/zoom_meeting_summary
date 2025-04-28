from flask import Flask
import socketio
# from flask_socketio import SocketIO, emit

import numpy as np
import threading
import time
import io
import whisper
from pydub import AudioSegment

# Initialize Flask app and SocketIO server
app = Flask(__name__)
sio = socketio.Server()

# Initialize Whisper model once
model = whisper.load_model("medium")

# Store the joined audio
audio_buffer = io.BytesIO()
buffer_size = 1024 * 1024  # 1MB buffer size
silence_threshold = 0.01  # Threshold for silence detection
min_audio_chunk_size = 1024  # Minimum chunk size to process

def is_not_silent(audio_chunk, threshold=silence_threshold):
    """Return True if audio chunk is not silent."""
    volume_norm = np.linalg.norm(audio_chunk) / len(audio_chunk)
    return volume_norm > threshold

def process_audio_data(audio_data):
    """Process audio data, e.g., transcribe, save, etc."""
    print(f"Processing {len(audio_data)} bytes of audio data")
    # Perform transcription using Whisper
    text = model.transcribe(audio_data)["text"]
    return text

@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
def mic_audio(sid, audio_data):
    """Receive audio data and accumulate"""
    print(f"Received {len(audio_data)} bytes of microphone audio")
    
    # Save to a .wav file (for simplicity, we're using a static filename)
    with open("received_audio.wav", "ab") as f:  # Open file in append-binary mode
        f.write(audio_data)
    
    sio.emit('server_response', {'message': 'Audio chunk saved'})

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    result = model.transcribe(audio_path)
    return result['text']

@app.route('/')
def index():
    return "WebSocket Server Running"

if __name__ == '__main__':
    # Set up the SocketIO server
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
    # Run the Flask app
    app.run(host='0.0.0.0', port=6000)

import socketio
import sounddevice as sd
import numpy as np
import threading

# Create Socket.IO client
sio = socketio.Client()

@sio.on('connect')
def on_connect():
    print('Connected to server')

@sio.on('server_response')
def on_server_response(data):
    print('Server says:', data)

@sio.on('disconnect')
def on_disconnect():
    print('Disconnected from server')

# Audio stream parameters
samplerate = 16000  # 16 kHz is common for speech
channels = 1
blocksize = 1024  # Small blocks for low latency

def audio_callback(indata, frames, time, status):
    """Callback to send mic data over socket"""
    if status:
        print(status)
    sio.emit('mic_audio', indata.tobytes())

def start_stream():
    """Start microphone stream"""
    with sd.InputStream(samplerate=samplerate, channels=channels, blocksize=blocksize, callback=audio_callback):
        threading.Event().wait()  # Wait forever

if __name__ == "__main__":
    sio.connect('http://localhost:6000')  # Update URL if needed
    start_stream()

import socketio
import sounddevice as sd
import numpy as np
import threading

# Create Socket.IO client
sio = socketio.Client()

samplerate = 16000
channels = 1
blocksize = 1024*160
silence_threshold = 0

@sio.on('connect')
def on_connect():
    print('Connected to server')

@sio.on('server_response')
def on_server_response(data):
    print('Transcription:', data['message'])

@sio.on('disconnect')
def on_disconnect():
    print('Disconnected from server')

def is_not_silent(audio_chunk, threshold=silence_threshold):
    volume_norm = np.linalg.norm(audio_chunk) / len(audio_chunk)
    return volume_norm > threshold

def audio_callback(indata, frames, time_info, status):
    if status:
        print('Audio stream error:', status)
    # print(indata.tobytes())
    # Check if the audio chunk is not silent
    # and send it to the server 
    if is_not_silent(indata):
        sio.emit('mic_audio', indata.tobytes())  # send raw PCM bytes (float32)

def start_stream():
    with sd.InputStream(samplerate=samplerate, channels=channels, blocksize=blocksize, dtype='float32', callback=audio_callback):
        threading.Event().wait()

if __name__ == "__main__":
    sio.connect('http://localhost:6000')
    start_stream()

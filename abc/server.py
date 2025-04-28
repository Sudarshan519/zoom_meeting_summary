from flask import Flask
import socketio
import io
import whisper
import numpy as np
import wave
import warnings
import time
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

model = whisper.load_model("medium")

audio_buffers = {}
buffer_threshold = 16000 *  2 * 4 # 3 seconds * 16000 samples/sec * 4 bytes per float32

sample_rate = 16000
channels = 1

@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")
    audio_buffers[sid] = io.BytesIO()

@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")
    audio_buffers.pop(sid, None)

@sio.event
def mic_audio(sid, data):
    """Receive raw PCM float32 audio chunks"""
    try:
        buffer = audio_buffers.get(sid)
        if buffer is None:
            buffer = io.BytesIO()
            audio_buffers[sid] = buffer

        buffer.write(data)

        if buffer.tell() >= buffer_threshold:
            print(f"Processing {buffer.tell()} bytes from {sid}")

            # Read bytes as float32 array
            buffer.seek(0)
            audio_np = np.frombuffer(buffer.read(), dtype=np.float32)

            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
            audio_int16 = (audio_np * 32767).astype(np.int16)

            # Save to proper WAV file
            temp_path = f"temp_{time.time()}.wav"
# temp_path = f"temp_.wav"
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
                buffer.seek(0)
                # wf.close()
            print(f"Saved audio to {temp_path}")
            # Transcribe
            result = model.transcribe(temp_path)
            transcription = result['text']
            print(f"Transcription: {transcription}")

            sio.emit('server_response', {'message': transcription}, room=sid)

            # Reset buffer
            buffer.seek(0)
            buffer.truncate()

    except Exception as e:
        print(f"Error during audio processing: {e}")
        sio.emit('server_response', {'message': 'Could not process audio'}, room=sid)

@app.route('/')
def index():
    return "WebSocket server is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)

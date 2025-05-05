# server.py
import asyncio
from flask import Flask
from openai import AsyncOpenAI
import socketio
import io
import whisper
import numpy as np
import wave
import warnings
import time
import os
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Create regular Flask app
flask_app = Flask(__name__)

# Create Async Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")

# Create ASGI app using Socket.IO's ASGIApp
app = socketio.ASGIApp(sio, flask_app)

model = whisper.load_model("small")
audio_buffers = {}
sample_rate = 16000
channels = 1
conversation = []
buffer_threshold = sample_rate * 4 * 5

aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def make_suggestion(text):
    print("🤖 Sending to GPT-4 for analysis...")
    prompt = "You're assisting a user during a meeting. Summarize the conversation and provide possible answers."
    conversation_text = "\n\n".join(conversation) + f"\n\n{text}"
    
    try:
        completion = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": conversation_text}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in make_suggestion: {e}")
        return "Sorry, I couldn't process that request."

async def transcribe_audio(sid, file_path):
    try:
        result = model.transcribe(file_path, fp16=False)
        return result['text'].strip()
    except Exception as e:
        print(f"[{sid}] Transcription error: {e}")
        return None
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@sio.event
async def connect(sid, environ):
    print(f"[+] Client connected: {sid}")
    audio_buffers[sid] = io.BytesIO()

@sio.event
async def disconnect(sid):
    print(f"[-] Client disconnected: {sid}")
    if sid in audio_buffers:
        del audio_buffers[sid]

@sio.event
async def mic_audio(sid, data):
    try:
        buffer = audio_buffers.get(sid, io.BytesIO())
        audio_buffers[sid] = buffer
        
        buffer.write(data)

        if buffer.tell() >= buffer_threshold:
            print(f"[{sid}] Processing {buffer.tell()} bytes")
            buffer.seek(0)
            
            audio_np = np.frombuffer(buffer.read(), dtype=np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)

            temp_path = f"temp_{sid}_{int(time.time())}.wav"
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            transcription = await transcribe_audio(sid, temp_path)
            if transcription:
                print(f"[{sid}] Transcription: {transcription}")
                suggestion = await make_suggestion(transcription)
                await sio.emit('server_response', {
                    'message': transcription,
                    'suggestion': suggestion
                }, room=sid)

            buffer.seek(0)
            buffer.truncate()

    except Exception as e:
        print(f"[{sid}] Error: {e}")
        await sio.emit('server_error', {'error': str(e)}, room=sid)

@flask_app.route('/')
def index():
    return "Whisper WebSocket Server is running!"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=3000,
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
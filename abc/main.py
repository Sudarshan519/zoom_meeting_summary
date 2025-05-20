# server.py
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import markdown
from openai import OpenAI
import socketio
import io
import whisper
import numpy as np
import wave
import warnings
import time
import os
import uvicorn
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Initialize FastAPI app
app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*") # Use AsyncServer for FastAPI
app.mount("/socket.io", socketio.ASGIApp(sio)) # Mount Socket.IO under a path

# Load Whisper model
# Consider loading the model once globally if it's thread-safe or use a separate process for transcription
model = whisper.load_model("small")

audio_buffers = {}
sample_rate = 16000
channels = 1
conversation = []
# 5 seconds of float32 (4 bytes/sample) for buffer threshold
buffer_threshold = sample_rate * 4 * 5

# Function to make suggestions using GPT-4o
async def makeSuggestion(text):
    print("🤖 Sending to GPT-4o for analysis...")
    prompt = "You're assisting a user during a meeting. Summarize the conversation and also provide the possible answers to question.Conversation [Question] [Answer] {Answer}"
    
    # Ensure conversation is managed carefully for concurrency if multiple users are interacting
    # For a single ongoing conversation, this is fine.
    conversation.append(text) # Add the latest transcription to the conversation
    conversation_text = "\n\n".join(conversation[-5:]) # Join the last few turns for context
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": conversation_text}
    ]

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response_content = completion.choices[0].message.content
        print(response_content)
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error generating suggestion."

@sio.event
async def connect(sid, environ):
    print(f"[+] Client connected: {sid}")
    audio_buffers[sid] = io.BytesIO()

@sio.event
async def disconnect(sid):
    print(f"[-] Client disconnected: {sid}")
    audio_buffers.pop(sid, None)

@sio.event
async def mic_audio(sid, data):
    try:
        buffer = audio_buffers.get(sid)
        if buffer is None:
            buffer = io.BytesIO()
            audio_buffers[sid] = buffer

        buffer.write(data)

        if buffer.tell() >= buffer_threshold:
            print(f"[{sid}] Processing {buffer.tell()} bytes")

            buffer.seek(0)
            audio_np = np.frombuffer(buffer.read(), dtype=np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)

            # Create a temporary WAV file
            temp_path = f"temp_{sid}_{int(time.time())}.wav"
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Perform transcription (this is CPU-bound and might block the event loop,
            # consider running in a separate thread if performance is critical)
            # FastAPI's run_in_threadpool can be used here for non-async functions.
            # Example: transcription_result = await run_in_threadpool(model.transcribe, temp_path, fp16=False)
            
            # For simplicity, keeping it direct for now.
            result = model.transcribe(temp_path, fp16=False)
            transcription = result['text'].strip()
            print(f"[{sid}] Transcription: {transcription}")
            
            await sio.emit('server_response', {'message': f"[{time.time()}] {transcription}"}, room=sid)
            
            # Get suggestion asynchronously
            suggestion = await makeSuggestion(transcription)
            await sio.emit('server_response_suggestion', {'message': "\n" + markdown.markdown(suggestion)}, room=sid)

            # Clean up temporary file
            os.remove(temp_path)
            
            # Reset buffer
            buffer.seek(0)
            buffer.truncate()

    except Exception as e:
        print(f"[{sid}] Error: {e}")
        await sio.emit('server_response', {'message': 'Error processing audio'}, room=sid)


from fastapi.templating import Jinja2Templates
# Configure Jinja2Templates to point to your templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    context = {
        "request": request, # Jinja2Templates requires the request object
        "title": "FastAPI Template Example",
        "name": "My Awesome App",
        # "current_date": datetime.date.today().strftime("%Y-%m-%d")
    }
    return templates.TemplateResponse("index.html",context)

    # return HTMLResponse("<h1>Whisper WebSocket Server is running!</h1>")

# To run this, you would typically use: uvicorn server:app --host 0.0.0.0 --port 3000 --reload
# The __name__ == '__main__' block for uvicorn is usually structured differently or omitted
# when running via the command line. However, for a single file runnable script:
# if __name__ == '__main__':
#     # You might want to run this in a separate terminal using the uvicorn command.
#     # The `uvicorn.run()` call will block the execution.
#     uvicorn.run(app, host="0.0.0.0", port=3000)
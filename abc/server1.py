# server.py
import threading
from flask import Flask
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
import eventlet
import eventlet.wsgi
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY" ),
)


warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

model = whisper.load_model("small")  # Change to "medium" or "large" if needed

audio_buffers = {}
sample_rate = 16000
channels = 1
conversation=[]
buffer_threshold = sample_rate * 4 * 5  # 3 seconds of float32 (4 bytes/sample)
def makeSuggestion(text):
    print("🤖 Sending to GPT-4 for analysis...")
    # You are an expert in communication analysis. Analyze this short transcript:
    # - Emotional tone
    prompt ="You're assisting a user during a meeting. Summarize the conversation and also provide the possible answers to question.Conversation [Question] [Answer] {Answer}"# f"""You are an bot helping user on meetin with context.Separate conversation and also suggest feasible solutions to help host for feasible solutions.Suggest latest answer at top.Respond like as if you are the host of meeting.Respond only in english"""
    conversation_text = "\n\n".join(conversation)  # Join the conversation list into a single string with line breaks
    conversation_text += f"\n\n{text}"
    # print(conversation_text)
    messages = [
        {"role": "system", "content":prompt},# "You are a coach helping actors improve their responses during interviews or auditions."},
        {"role": "user", "content": conversation_text}
    ]

    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=messages,
    #     temperature=0.7,
    #     max_tokens=500
    # )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
        #   [
        #     {"role": "developer", "content": "Talk like a pirate."},
        #     {
        #         "role": "user",
        #         "content": "How do I check if a Python object is an instance of a class?",
        #     },
        # ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

@sio.event
def connect(sid, environ):
    print(f"[+] Client connected: {sid}")
    audio_buffers[sid] = io.BytesIO()

@sio.event
def disconnect(sid):
    print(f"[-] Client disconnected: {sid}")
    audio_buffers.pop(sid, None)

def transcribe_audio(sid, file_path):
    try:
        result = model.transcribe(file_path, fp16=False)
        transcription = result['text'].strip()
        print(f"[{sid}] Transcription: {transcription}")

        sio.emit('server_response', {'message':f"[{time.time()}]" +transcription}, room=sid)
    except Exception as e:
        print(f"[{sid}] Transcription error: {e}")
        sio.emit('server_response', {'message': 'Transcription error'}, room=sid)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
@sio.event
def mic_audio(sid, data):
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

            temp_path = f"temp_{sid}_{int(time.time())}.wav"
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
        # Step 3: Start transcription in a separate thread
            # threading.Thread(target=transcribe_audio, args=(sid, temp_path, result)).start()

            result = model.transcribe(temp_path, fp16=False)
            transcription = result['text'].strip()
            print(f"[{sid}] Transcription: {transcription}")
            sio.emit('server_response', {'message': transcription}, room=sid)
            suggestion=makeSuggestion(transcription)
            # suggestion = ''
            sio.emit('server_response_suggestion', {'message':" \n"+ markdown.markdown(suggestion)}, room=sid)

            os.remove(temp_path)
            buffer.seek(0)
            buffer.truncate()

    except Exception as e:
        print(f"[{sid}] Error: {e}")
        sio.emit('server_response', {'message': 'Error processing audio'}, room=sid)

@app.route('/')
def index():
    return "Whisper WebSocket Server is running!"

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 3000)), app,debug=True)

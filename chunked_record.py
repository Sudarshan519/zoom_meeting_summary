import requests
import sounddevice as sd
import soundfile as sf
import queue
import threading
import time
import os
from dotenv import load_dotenv
import os
# Tweakable Parameters
SILENCE_THRESHOLD = 0.01  # RMS energy below this = silence
SILENCE_DURATION = 1.5    # seconds of silence to consider a break (speech finished)
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

audio_accumulation_buffer = []
conversation_buffer=""

def is_really_silent(audio):
    """Return True if audio chunk is very low energy (silent)"""
    rms = np.sqrt(np.mean(np.square(audio)))
    return rms < SILENCE_THRESHOLD

# Load environment variables from the .env file
load_dotenv()
gemini_key=os.environ.get("GEMINI_API_KEY")
import numpy as np
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key"
WHISPER_API_KEY= os.getenv("WHISPER_API_KEY") or "your-whisper-api-key"
ANALYSIS_LOG = "transcript/analysis_log.txt"
TRANSCRIPT_LOG = "transcript/transcript_log.txt"
import whisper
model = whisper.load_model("base")
def transcribeWhisper(file):
    name=f'audio.webm'
    with open(name, "wb") as f:
        file.save(f)
        result = model.transcribe(file.name)
        print(result['text'])
    return result['text']
def transcribe_chunk(file_path):
    global conversation_buffer
    name=f'audio.webm'
    result = model.transcribe(file_path)
    conversation_buffer+=result['text']
    print(result['text'])
    return conversation_buffer
    print(f"🧠 Transcribing {file_path}...")
    with open(file_path, "rb") as audio_file:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": audio_file},
            data={"model": "whisper-1"}
        )
    return response.json().get("text", "")

def analyze_transcript(text):
        print("🤖 Sending to GPT-4 for analysis...")
        gemini_url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_key}"
        history = [
                        {"role": "user" , "parts":[{"text": text}]}
                 
                    ]
        body_data={
                            "contents":history,
                            "systemInstruction": {
                                "role": "user",
                                "parts": [
                                {
                            "text":"Help user guide the meeting with answersProvide me response as if you are a responding to question "
                                }
                                ]
                            },
                            "generationConfig": {
                                "temperature": 0.15,
                                "topK": 40,
                                "topP": 0.95,
                                "maxOutputTokens": 8192,
                                "responseMimeType": "text/plain"
                            }
                            }
        ai_response = requests.post(gemini_url,json=body_data)
        return ai_response.json()['candidates'][0]['content']['parts'][0]['text']

# You are an expert in communication analysis. Analyze this short transcript:
# - Emotional tone
#     prompt = f"""
# You are an bot helping user on meetin with context.Separate conversation and also suggest feasible solutions to help host for feasible solutions.
# Also separate the conversation with the host and users and the bot.
# Here's the transcript:  
# {text}

# Provide:
# - Key takeaways
# - Suggestions for context
# """
#     response = requests.post(
#         "https://api.openai.com/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {OPENAI_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": "gpt-4",
#             "messages": [{"role": "user", "content": "Provide user with suggestions"}]
#         }
#     )
#     return response.json()['choices'][0]['message']['content']
    
# ========== SETTINGS ==========
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 2  # seconds
OUTPUT_DIR = "audio_chunks"
SILENCE_THRESHOLD = 0.01  # Adjust for mic sensitivity
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
q = queue.Queue()
chunk_count = 0
is_recording = True

def audio_callback(indata, frames, time_info, status):
    """Add audio chunks to the queue."""
    q.put(indata.copy())

def is_silent(audio_data):
    """Check if audio chunk is silent using RMS."""
    rms = np.sqrt(np.mean(audio_data**2))
    return rms < SILENCE_THRESHOLD
def contains_question(text):
    """Detects if the text contains a question."""
    question_words = ["who", "what", "when", "where", "why", "how", "which", "do", "does", "did", "can", "could", "would", "should", "is", "are", "am", "will", "shall", "have",'Please','Can you','Could you','Would you','Should I','Do I','Is it possible','Is there any chance']
    lines = text.lower().splitlines()

    for line in lines:
        line = line.strip()
        if line.endswith("?"):
            return True
        for word in question_words:
            if line.startswith(word + " ") or f" {word} " in line:
                return True
    return False
def analyze_in_background(transcript, chunk_num):
    """Background task: analyze transcript separately."""
    if True:
        analysis = analyze_transcript(transcript)
        print(f"🧠 Analysis for chunk {chunk_num}:\n{analysis}")
        with open(ANALYSIS_LOG, "a") as log:
            log.write(f"\n--- Chunk {chunk_num} ---\n")
            log.write(f"Analysis:\n{analysis}\n\n")
        print(f"✅ Analysis for chunk {chunk_num} written to log.")
    else:
        print("❌ No question detected. Skipping analysis.")

def process_transcription(filename, chunk_num):
    """Background task: transcribe audio chunk and start analysis if needed."""
    transcript = transcribe_chunk(filename)
    if transcript.strip():
        with open(TRANSCRIPT_LOG, "a") as log:
            log.write(f"\n--- Chunk {chunk_num} ---\n")
            log.write(f"Transcript:\n{transcript}\n\n")
        print(f"✅ Transcript for chunk {chunk_num} saved to log.")

        # 🔥 Now spawn a thread ONLY for analysis
        analysis_thread = threading.Thread(target=analyze_in_background, args=(transcript, chunk_num))
        analysis_thread.daemon = True
        analysis_thread.start()

    else:
        print("⚠️ Empty transcript. Skipping.")
last_active_time = time.time()
def write_chunks():
    """Process audio chunks, detect real speech pauses and save intelligently."""
    global chunk_count
    global last_active_time
    global audio_accumulation_buffer
    buffer = []
    start_time = time.time()

    while is_recording or not q.empty():
        try:
            data = q.get(timeout=1)
            buffer.append(data)

            if time.time() - start_time >= CHUNK_DURATION:
                combined = np.concatenate(buffer)

                if not is_really_silent(combined):
                    audio_accumulation_buffer.append(combined)
                    last_active_time = time.time()
                    print("🎙️ Active speech... accumulating.")

                buffer = []
                start_time = time.time()

                # Check for silence
                silence_time = time.time() - last_active_time
                if silence_time >= SILENCE_DURATION and audio_accumulation_buffer:
                    final_audio = np.concatenate(audio_accumulation_buffer)

                    filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count}.wav")
                    with sf.SoundFile(filename, mode='w', samplerate=SAMPLE_RATE,
                                      channels=CHANNELS, subtype='PCM_16') as f:
                        f.write(final_audio)
                    print(f"💾 Saved natural speech chunk {chunk_count} → {filename}")

                    # 🔥 Start transcription in separate thread
                    t = threading.Thread(target=process_transcription, args=(filename, chunk_count))
                    t.daemon = True
                    t.start()

                    chunk_count += 1
                    audio_accumulation_buffer.clear()

        except queue.Empty:
            continue
def contains_question_gpt(text):
    prompt = f"Does this text contain any questions?\n\n{text}\n\nRespond with 'yes' or 'no'."
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    )
    answer = response.json()['choices'][0]['message']['content'].strip().lower()
    return "yes" in answer

# 🎧 Simulated Recording Thread
def record_audio(duration=None):
    global is_recording
    print("🎙️ Recording started (Ctrl+C to stop)...")
    writer_thread = threading.Thread(target=write_chunks)
    writer_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        try:
            if duration:
                sd.sleep(duration * 1000)
                is_recording = False
            else:
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            print("🛑 Stopping recording...")
            is_recording = False
            writer_thread.join()

def start_recording(duration=None):
    """Start live recording with silence skipping."""
    global is_recording

    print("🎙️ Recording started (Ctrl+C to stop)...")
    writer_thread = threading.Thread(target=write_chunks)
    writer_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        try:
            if duration:
                sd.sleep(duration * 1000)
                is_recording = False
            else:
                while True:
                    time.sleep(0.1)
                    # sd.sleep(1000)
        except KeyboardInterrupt:
            print("🛑 Stopping recording...")
            is_recording = False
            writer_thread.join()

if __name__ == "__main__":
    start_recording()

import requests
import sounddevice as sd
import soundfile as sf
import queue
import threading
import time
import os
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
import numpy as np
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key"
WHISPER_API_KEY= os.getenv("WHISPER_API_KEY") or "your-whisper-api-key"
ANALYSIS_LOG = "transcript/analysis_log.txt"
TRANSCRIPT_LOG = "transcript/transcript_log.txt"


def transcribe_chunk(file_path):
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
# You are an expert in communication analysis. Analyze this short transcript:
# - Emotional tone
    prompt = f"""
You are an bot helping user on meetin with context.Separate conversation and also suggest feasible solutions to help host for feasible solutions.
Also separate the conversation with the host and users and the bot.
Here's the transcript:  
{text}

Provide:
- Key takeaways
- Suggestions for context
"""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    return response.json()['choices'][0]['message']['content']
    
# ========== SETTINGS ==========
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 10  # seconds
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
    question_words = ["who", "what", "when", "where", "why", "how", "which", "do", "does", "did", "can", "could", "would", "should", "is", "are", "am", "will", "shall", "have"]
    lines = text.lower().splitlines()

    for line in lines:
        line = line.strip()
        if line.endswith("?"):
            return True
        for word in question_words:
            if line.startswith(word + " ") or f" {word} " in line:
                return True
    return False

def write_chunks():
    """Process audio chunks from the queue and write to file if not silent."""
    global chunk_count
    buffer = []
    start_time = time.time()

    while is_recording or not q.empty():
        try:
            data = q.get(timeout=1)
            buffer.append(data)

            # Save every CHUNK_DURATION seconds
            if time.time() - start_time >= CHUNK_DURATION:
                combined = np.concatenate(buffer)

                if is_silent(combined):
                    print("🔇 Skipped silent chunk")
                else:
                    filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count}.wav")
                    with sf.SoundFile(filename, mode='w', samplerate=SAMPLE_RATE,
                                      channels=CHANNELS, subtype='PCM_16') as f:
                        f.write(combined)
                    print(f"💾 Saved chunk {chunk_count} → {filename}")
                    chunk_count += 1

                 # 🔍 Transcribe the chunk
                    transcript = transcribe_chunk(filename)
                    if transcript.strip():
                        # Save the transcript to a log file (all transcripts)
                        with open(TRANSCRIPT_LOG, "a") as log:
                            log.write(f"\n--- Chunk {chunk_count} ---\n")
                            log.write(f"Transcript:\n{transcript}\n\n")

                        print(f"✅ Transcript for chunk {chunk_count} saved to log.")

                        # If there's a question in the transcript, analyze it
                        if contains_question(transcript):
                            analysis = analyze_transcript(transcript)

                            # Append analysis to the log file
                            with open(ANALYSIS_LOG, "a") as log:
                                log.write(f"\n--- Chunk {chunk_count} ---\n")
                                log.write(f"Analysis:\n{analysis}\n\n")

                            print(f"✅ Analysis for chunk {chunk_count} written to log.")
                        else:
                            print("❌ No question detected. Skipping analysis.")
                    else:
                        print("⚠️ Empty transcript. Skipping.")

                    chunk_count += 1

                buffer = []
                start_time = time.time()

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

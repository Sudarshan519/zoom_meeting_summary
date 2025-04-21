import threading
import time
import queue
import numpy as np
import requests
import soundfile as sf
import os
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
import numpy as np
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key"
# Constants
CHUNK_DURATION = 5  # seconds
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_DIR = "chunks"
TRANSCRIPT_LOG = "transcripts.txt"
ANALYSIS_LOG = "analysis.txt"

# Shared queues and flags
q_audio = queue.Queue()
q_transcribe = queue.Queue()
is_recording = True
chunk_count = 0

# Stub functions
def is_silent(audio_data):
    return np.abs(audio_data).mean() < 0.01

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

def contains_question(text):
    return "?" in text

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
   
# 🎙️ Audio Processing Thread
def write_chunks():
    global chunk_count
    buffer = []
    start_time = time.time()

    while is_recording or not q_audio.empty():
        try:
            data = q_audio.get(timeout=1)
            buffer.append(data)

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

                    # Send filename for transcription
                    q_transcribe.put((chunk_count, filename))

                    chunk_count += 1

                buffer = []
                start_time = time.time()

        except queue.Empty:
            continue

# ✍️ Transcription + Analysis Thread
def transcribe_and_analyze():
    while is_recording or not q_transcribe.empty():
        try:
            chunk_id, filename = q_transcribe.get(timeout=1)
            transcript = transcribe_chunk(filename)

            if transcript.strip():
                with open(TRANSCRIPT_LOG, "a") as log:
                    log.write(f"\n--- Chunk {chunk_id} ---\n")
                    log.write(f"Transcript:\n{transcript}\n\n")

                print(f"✅ Transcript for chunk {chunk_id} saved to log.")

                if contains_question(transcript):
                    analysis = analyze_transcript(transcript)
                    with open(ANALYSIS_LOG, "a") as log:
                        log.write(f"\n--- Chunk {chunk_id} ---\n")
                        log.write(f"Analysis:\n{analysis}\n\n")

                    print(f"✅ Analysis for chunk {chunk_id} written to log.")
                else:
                    print("❌ No question detected. Skipping analysis.")
            else:
                print("⚠️ Empty transcript. Skipping.")

        except queue.Empty:
            continue

# 🎧 Simulated Recording Thread
def record_audio():
    global is_recording
    for _ in range(1150):  # Simulate 50 chunks
        dummy_data = np.random.randn(int(SAMPLE_RATE * 0.1))
        q_audio.put(dummy_data)
        time.sleep(0.1)
    is_recording = False

# 🧵 Run all threads
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    record_thread = threading.Thread(target=record_audio)
    chunk_writer_thread = threading.Thread(target=write_chunks)
    transcriber_thread = threading.Thread(target=transcribe_and_analyze)

    record_thread.start()
    chunk_writer_thread.start()
    transcriber_thread.start()

    record_thread.join()
    chunk_writer_thread.join()
    transcriber_thread.join()

    print("🎉 All recording, processing, and analysis completed.")

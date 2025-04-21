import requests
import sounddevice as sd
import soundfile as sf
import queue
import threading
import time
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables from the .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key"
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY") or "your-whisper-api-key"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "your-gemini-api-key"
ANALYSIS_LOG = f"transcript/analysis_log{time.time()}.txt"
TRANSCRIPT_LOG = f"transcript/transcript_log{time.time()}.txt"

# Ensure transcript directory exists
os.makedirs(os.path.dirname(TRANSCRIPT_LOG), exist_ok=True)

def transcribe_chunk(file_path):
    print(f"🧠 Transcribing {file_path}...")
    try:
        with open(file_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": audio_file},
                data={"model": "whisper-1"},
                timeout=30  # Increased timeout
            )
        return response.json().get("text", "")
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return ""

def analyze_transcript(text):
    if not text.strip():
        return "No content to analyze"
    
    print("🤖 Sending to GPT-4 for analysis...")
    with open(TRANSCRIPT_LOG, "r") as log:
        text=log.read()
    nprompt=   f"""
Answers to Participant Questions

Make your answers helpful and full-featured — like a professional assistant or project manager. Where relevant, link or recommend resources/tools. Avoid vague responses.
if more search needed provide [search] and [tools] to find complete answers.
Transcript:
{text}

Provide concise, actionable output.
Analyze last contents of transcript.
"""
    prompt = f"""
🧠 Prompt: Meeting Assistant Bot
You are an AI assistant helping users during or after meetings by analyzing conversation and provide some technical feedaback for solving issues on web platform and providing context-aware insights and solutions. Analyze the following conversation transcript according to the instructions below.

🔹 Instructions:
Speaker Identification:
Separate the conversation by speakers if possible. Use names or roles (e.g., Host, Participant 1, etc.) if identified.

Questions & Action Items:
Identify all questions asked by participants. Highlight any action items (e.g., tasks, follow-ups, deliverables).

Key Takeaways:
Summarize the core discussion points, decisions made, and significant observations.

Suggestions for the Host:
Suggest feasible next steps or solutions for the meeting host or leader, based on the conversation context.

Answer Participant Questions:
Provide clear and detailed answers to all questions asked by participants, using available context. If necessary, recommend web search or external tools to find complete answers.

📌 Output Format:
Respond in structured bullet points under clear headers like:

Speakers

Questions & Action Items

Key Takeaways

Suggestions for the Host

Answers to Participant Questions

Make your answers helpful, context-rich, and full-featured — like a professional assistant or project manager. Where relevant, link or recommend resources/tools. Avoid vague responses.
Transcript:
{text}

Provide concise, actionable output.
Provide me response for last question analyzing full conversation.
"""
    try:
        gemini_url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        body_data={
                            "contents":            [{"role": "user", "parts": [{"text": nprompt}]}]
,
                            "systemInstruction": {
                                "role": "user",
                                "parts": [
                                {
                            "text":"You are an AI assistant helping users during or after meetings by analyzing conversation transcripts and providing context-aware insights and solutions. Analyze the following conversation transcript according to the instructions below.Provide response for host with feasible and helpful solutions during meetings"
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
        # if ai_response.status_code == 200:
        # print(ai_response.json())
        result=ai_response.json()['candidates'][0]['content']['parts'][0]['text']
        print(result)
        return result
     
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=30  # Increased timeout
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"⚠️ Analysis error: {e}")
        return "Analysis failed"

# ========== SETTINGS ==========
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 10  # seconds
OUTPUT_DIR = "audio_chunks"
SILENCE_THRESHOLD = 0.01  # Adjust for mic sensitivity
MIN_CHUNK_LENGTH = 2.0  # Minimum duration to process (seconds)
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
audio_queue = queue.Queue()
chunk_count = 0
is_recording = True

def audio_callback(indata, frames, time_info, status):
    """Add audio chunks to the queue."""
    audio_queue.put(indata.copy())

def is_silent(audio_data):
    """Check if audio chunk is silent using RMS."""
    if len(audio_data) == 0:
        return True
    rms = np.sqrt(np.mean(audio_data**2))
    return rms < SILENCE_THRESHOLD

def contains_question(text):
    """Detects if the text contains a question."""
    if not text.strip():
        return False
        
    question_words = ["who", "what", "when", "where", "why", "how", "which", 
                     "do", "does", "did", "can", "could", "would", "should", 
                     "is", "are", "am", "will", "shall", "have", "?"]
    
    text_lower = text.lower()
    return any(q_word in text_lower for q_word in question_words)

def process_chunk(chunk_data, chunk_num):
    """Process and analyze a single audio chunk."""
    if len(chunk_data) < SAMPLE_RATE * MIN_CHUNK_LENGTH:
        print(f"🔇 Skipped short chunk ({len(chunk_data)/SAMPLE_RATE:.1f}s)")
        return
        
    filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_num}.wav")
    try:
        sf.write(filename, chunk_data, SAMPLE_RATE)
        print(f"💾 Saved chunk {chunk_num} → {filename}")
        
        transcript = transcribe_chunk(filename)
        print(transcript)
        if not transcript.strip():
            print("⚠️ Empty transcript")
            return
            
        # Save transcript
        with open(TRANSCRIPT_LOG, "a", encoding='utf-8') as log:
            log.write(f"\n--- Chunk {chunk_num} ---\n{transcript}\n\n")
        print(f"✅ Transcript saved for chunk {chunk_num}")
        
        # Analyze if contains question
        if contains_question(transcript):
            analysis = analyze_transcript(transcript)
            with open(ANALYSIS_LOG, "a", encoding='utf-8') as log:
                log.write(f"\n--- Chunk {chunk_num} ---\n{analysis}\n\n")
            print(f"📝 Analysis saved for chunk {chunk_num}")
        else:
            print("❌ No question detected")
            
    except Exception as e:
        print(f"⚠️ Chunk processing error: {e}")

def write_chunks():
    """Process audio chunks from the queue."""
    global chunk_count
    buffer = []
    last_chunk_time = time.time()
    
    while is_recording or not audio_queue.empty():
        try:
            # Get audio data with timeout
            data = audio_queue.get(timeout=1)
            buffer.append(data)
            
            # Process if we have enough audio or recording stopped
            current_time = time.time()
            if (current_time - last_chunk_time >= CHUNK_DURATION) or (not is_recording and buffer):
                if buffer:
                    chunk_data = np.concatenate(buffer)
                    if not is_silent(chunk_data):
                        process_chunk(chunk_data, chunk_count)
                        chunk_count += 1
                    else:
                        print("🔇 Skipped silent chunk")
                        
                    buffer = []
                last_chunk_time = current_time
                
        except queue.Empty:
            continue

def start_recording(duration=None):
    """Start live recording."""
    global is_recording
    
    print("🎙️ Recording started (Press Ctrl+C to stop)...")
    writer_thread = threading.Thread(target=write_chunks, daemon=True)
    writer_thread.start()
    
    start_time = time.time()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                          callback=audio_callback, blocksize=SAMPLE_RATE//10):
            if duration:
                while time.time() - start_time < duration:
                    time.sleep(0.1)
            else:
                while True:
                    time.sleep(1)
                    
    except KeyboardInterrupt:
        print("\n🛑 Stopping recording...")
    finally:
        is_recording = False
        writer_thread.join(timeout=5)
        print("🎚️ Recording stopped")

if __name__ == "__main__":
    start_recording()
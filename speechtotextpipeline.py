import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
# from openai import OpenAI
# client = OpenAI()
# client.api_key= "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"
import os
import threading

# ================= CONFIG =================
AUDIO_MIC = "mic_recording.wav"
AUDIO_SPK = "speaker_recording.wav"
AUDIO_COMBINED = "combined_audio.wav"
TRANSCRIPT_FILE = "transcript.txt"
ANALYSIS_FILE = "analysis.txt"
RECORD_DURATION = 20  # seconds
SAMPLE_RATE = 44100
# client.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"
# ==========================================

# Optional: Choose specific devices (or leave None for default)
mic_device = None
speaker_device = None

# ========= 1. Record Mic & Speaker =========
mic_data = []
speaker_data = []

def record_mic():
    print("🎙️ Recording mic...")
    mic_stream = sd.InputStream(device=mic_device, channels=1, samplerate=SAMPLE_RATE,
                                callback=lambda indata, frames, time, status: mic_data.append(indata.copy()))
    with mic_stream:
        sd.sleep(RECORD_DURATION * 1000)

def record_speaker():
    print("🔊 Recording speaker (loopback)...")
    speaker_stream = sd.InputStream(device=speaker_device, channels=2, samplerate=SAMPLE_RATE,
                                    callback=lambda indata, frames, time, status: speaker_data.append(indata.copy()))
    with speaker_stream:
        sd.sleep(RECORD_DURATION * 1000)

def record_audio():
    mic_thread = threading.Thread(target=record_mic)
    # speaker_thread = threading.Thread(target=record_speaker)
    mic_thread.start()
    # speaker_thread.start()
    mic_thread.join()
    # speaker_thread.join()
    try:
    # Save both recordings
        sf.write(AUDIO_MIC, np.concatenate(mic_data), SAMPLE_RATE)
        # sf.write(AUDIO_SPK, np.concatenate(speaker_data), SAMPLE_RATE)
        print(f"✅ Saved mic to {AUDIO_MIC}, speaker to {AUDIO_SPK}")
    except Exception as e:
        print(e)
# ========= 2. Merge Audio Tracks (Optional) =========
def combine_audio_files(file1, file2, output_file):
    print("🎚️ Merging mic and speaker audio...")
    data1, sr1 = sf.read(file1, always_2d=True)
    data2, sr2 = sf.read(file2, always_2d=True)

    if sr1 != sr2:
        raise ValueError("Sample rates do not match")

    # Convert to mono by averaging channels
    data1 = np.mean(data1, axis=1, keepdims=True)
    data2 = np.mean(data2, axis=1, keepdims=True)

    # Pad shorter track
    len_diff = len(data1) - len(data2)
    if len_diff > 0:
        data2 = np.pad(data2, ((0, len_diff), (0, 0)), mode='constant')
    elif len_diff < 0:
        data1 = np.pad(data1, ((0, -len_diff), (0, 0)), mode='constant')

    # Mix audio
    mixed = (data1 + data2) / 2
    sf.write(output_file, mixed, sr1)
    print(f"🎵 Combined audio saved to {output_file}")
# ========= 3. Transcribe with Whisper =========
def transcribe_audio(file_path):
    import os
    import requests

    api_key = os.getenv("WHISPER_AI_KEY") or "your-api-key"
    audio_file_path = 'combined_audio.wav'  # Path to the audio file you want to transcribe
    model = 'whisper-1'
    language = 'en'

    headers = {
        'Authorization': f'Bearer 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'
    }

    with open(audio_file_path, 'rb') as audio_file:
        files = {
            'file': (audio_file_path, audio_file, 'audio/wav')
        }
        data = {
            'model': model,
            'language': language
        }
        response = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, files=files, data=data)

    print(response.json())
    with open(TRANSCRIPT_FILE, "w") as f:
        f.write(response.json()['text']+"\n")
    print(f"📝 Transcript saved to {TRANSCRIPT_FILE}")

# ========= 4. Analyze Transcript =========
def analyze_transcript(text):
    print("🤖 Sending to GPT for analysis...")
    prompt = f"""
You are an expert in conversation analysis. Here's the transcript:

{text}

Please extract:
- Key takeaways
- Issues or concerns
- Suggestions for improvement
"""
    headers = {
        'Authorization': f'Bearer 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'
    }

    data = {
             "model": "gpt-4o", 
             "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
                ]

                    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers,   json=data)

    print(response.json())
    analysis = response.json()['choices'][0]['message']['content']
    with open(ANALYSIS_FILE, "w") as f:
        f.write(analysis+"\n")
    print(f"💡 Analysis saved to {ANALYSIS_FILE}")
    return analysis

# ========= 5. Full Pipeline =========
def run_full_pipeline():
    record_audio()
    combine_audio_files(AUDIO_MIC, AUDIO_SPK, AUDIO_COMBINED)
    transcript = transcribe_audio(AUDIO_COMBINED)
    suggestions = analyze_transcript(transcript)
    print("\n✨ Final Suggestions:\n", suggestions)

if __name__ == "__main__":
    while True:

        run_full_pipeline()

import os
import requests

api_key = os.getenv("WHISPER_AI_KEY") or "your-api-key"
print(api_key)
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

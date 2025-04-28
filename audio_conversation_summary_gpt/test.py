import requests

# Replace this path with an actual audio file (e.g., a real .webm or .wav file)
file_path = "chunk_34.wav"

# Send the file to your Flask backend
with open(file_path, "rb") as f:
    files = {"file": ("test.webm", f, "audio/webm")}
    response = requests.post("http://127.0.0.1:5000/transcribe", files=files)

# Print the response from the backend
print("Status Code:", response.status_code)
print("Response:", response.json())

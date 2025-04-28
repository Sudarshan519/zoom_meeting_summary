import speech_recognition as sr
import pyaudio

# Set up the recognizer
recognizer = sr.Recognizer()

# You will need to set the correct index for the virtual microphone device
# Use the following code to find the correct index of the virtual microphone (e.g., VB-Audio or Soundflower)
def list_audio_devices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(f"Device {i}: {p.get_device_info_by_index(i)['name']}")

# Uncomment this to list devices and find the correct index
# list_audio_devices()

# Set up the virtual audio device (your virtual microphone, e.g., VB-Audio or Soundflower)
mic_index = 1  # Change this index according to your system setup

# Use the virtual microphone as the input device
mic = sr.Microphone(device_index=mic_index)

# Function to capture and transcribe system audio in real-time
def capture_and_transcribe_system_audio():
    with mic as source:
        print("Listening to system audio...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        while True:
            try:
                # Capture audio from the virtual microphone
                audio = recognizer.listen(source)
                print("Transcribing system audio...")
                
                # Use Google's API to transcribe the captured audio
                transcription = recognizer.recognize_google(audio)  # You can also use Sphinx for offline
                print(f"Transcribed text: {transcription}")
            
            except sr.UnknownValueError:
                print("Sorry, I could not understand the system audio.")
            except sr.RequestError as e:
                print(f"Request error: {e}")

# Start the real-time transcription of system audio
capture_and_transcribe_system_audio()

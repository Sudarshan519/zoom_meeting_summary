import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time

# Configs
duration = 20  # seconds
mic_device = None         # Default input (you can set a specific device index)
speaker_device = None     # Default output loopback (some systems support this)

# Check available devices
print(sd.query_devices())

# Replace speaker_device with your loopback device if available
# On Windows, WASAPI loopback can be used
try:
    # Windows WASAPI loopback example
    speaker_device_info = sd.query_devices(kind='output')
    speaker_device = speaker_device_info['index']
except Exception as e:
    print("Could not detect loopback device:", e)

# Buffers to hold recorded audio
mic_data = []
speaker_data = []

def record_mic():
    print("Recording mic...")
    mic = sd.InputStream(device=mic_device, channels=1, samplerate=44100, callback=lambda indata, frames, time, status: mic_data.append(indata.copy()))
    with mic:
        sd.sleep(int(duration * 1000))

def record_speaker():
    print("Recording speaker...")
    speaker = sd.InputStream(device=speaker_device, channels=2, samplerate=44100, callback=lambda indata, frames, time, status: speaker_data.append(indata.copy()), dtype='float32', blocksize=1024)
    with speaker:
        sd.sleep(int(duration * 1000))

# Start both recordings in parallel
mic_thread = threading.Thread(target=record_mic)
speaker_thread = threading.Thread(target=record_speaker)

mic_thread.start()
speaker_thread.start()

mic_thread.join()
speaker_thread.join()

# Save audio
sf.write('mic_recording.wav', np.concatenate(mic_data), 44100)
sf.write('speaker_recording.wav', np.concatenate(speaker_data), 44100)

print("Done recording!")

import asyncio
import io
import os
import time
import warnings
import wave
import collections

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import markdown
import uvicorn
import whisper

from fastapi.concurrency import run_in_threadpool

# --- Initialize API Client and Load Model ---
load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
model = whisper.load_model("small") # Consider 'base' or 'small' for speed vs accuracy
# ---------------------------------------------

app = FastAPI()

# --- Audio Processing Constants ---
sample_rate = 16000
channels = 1
bytes_per_sample = 4 # float32 uses 4 bytes per sample

# Adjusted for more "live" feel
# Client sends 1-second chunks (from audio-processor.js sendInterval)
# Server accumulates a larger window for Whisper context, but transcribes frequently
TRANSCRIBE_WINDOW_SECONDS = 5    # The total duration of audio Whisper will analyze
TRANSCRIBE_STRIDE_SECONDS = 1.5  # How often to trigger a transcription (e.g., every 1.5 seconds)
VAD_BUFFER_SECONDS = 2           # Additional buffer after voice ends for VAD in client
MIN_AUDIO_CHUNK_SIZE = int(sample_rate * bytes_per_sample * TRANSCRIBE_STRIDE_SECONDS)

# Buffers and Histories (per client, per source)
audio_buffers = {
    "mic": collections.deque(maxlen=int(sample_rate * TRANSCRIBE_WINDOW_SECONDS)),
    "tab": collections.deque(maxlen=int(sample_rate * TRANSCRIBE_WINDOW_SECONDS))
}
last_transcription_time = {
    "mic": 0,
    "tab": 0
}
# Keep track of previously sent text to identify new parts
previous_full_transcription = {
    "mic": "",
    "tab": ""
}
conversation_history = {
    "mic": {},
    "tab": {}
}

# -----------------------------------------------

# --- OpenAI Suggestion Function ---
async def make_suggestion(client_id: str, source_type: str, text: str):
    print(f"🤖 Sending to GPT-4o for analysis ({source_type}) for {client_id}...")
    prompt = "You're assisting a user during a meeting. Summarize the conversation and also provide possible answers to questions. Format the answer as a markdown list or paragraph.Conversation: [Question] [Answer] {Answer}"

    if client_id not in conversation_history[source_type]:
        conversation_history[source_type][client_id] = []
    conversation_history[source_type][client_id].append(text)

    context_text = "\n\n".join(conversation_history[source_type][client_id][-5:])

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context_text}
    ]

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response_content = completion.choices[0].message.content
        print(f"GPT-4o response ({source_type}) for {client_id}: {response_content}")
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API ({source_type}) for {client_id}: {e}")
        return "Error generating suggestion."

# --- WebSocket Processing Handler (reusable for both mic and tab) ---
async def handle_audio_websocket(websocket: WebSocket, source_type: str):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"[+] Client {client_id} connected for {source_type} audio.")

    # Initialize client-specific buffers and histories if not already present
    if client_id not in conversation_history[source_type]:
        conversation_history[source_type][client_id] = []
    if client_id not in audio_buffers[source_type]:
        audio_buffers[source_type][client_id] = collections.deque(maxlen=int(sample_rate * TRANSCRIBE_WINDOW_SECONDS))
    if client_id not in last_transcription_time[source_type]:
        last_transcription_time[source_type][client_id] = time.time()
    if client_id not in previous_full_transcription[source_type]:
        previous_full_transcription[source_type][client_id] = ""

    try:
        while True:
            audio_data_bytes = await websocket.receive_bytes()
            # Convert bytes to numpy array of float32
            current_audio_np = np.frombuffer(audio_data_bytes, dtype=np.float32)

            # Append to rolling buffer
            audio_buffers[source_type][client_id].extend(current_audio_np)

            current_time = time.time()

            # Only transcribe if enough time has passed and we have enough data
            if (current_time - last_transcription_time[source_type][client_id] >= TRANSCRIBE_STRIDE_SECONDS and
                len(audio_buffers[source_type][client_id]) >= int(sample_rate * TRANSCRIBE_STRIDE_SECONDS)):

                last_transcription_time[source_type][client_id] = current_time

                # Get the current window of audio from the deque
                audio_for_whisper = np.array(audio_buffers[source_type][client_id], dtype=np.float32)

                if audio_for_whisper.size == 0 or np.all(audio_for_whisper == 0):
                    # print(f"[{client_id}][{source_type}] Skipping transcription for silent/empty chunk.")
                    continue # Skip if buffer is empty or silent

                try:
                    transcription_result = await run_in_threadpool(
                        model.transcribe, audio_for_whisper, fp16=False
                    )
                    full_transcription = transcription_result['text'].strip()

                    # Logic to find the "new" part of the transcription
                    new_text = ""
                    if full_transcription.startswith(previous_full_transcription[source_type][client_id]):
                        new_text = full_transcription[len(previous_full_transcription[source_type][client_id]):].strip()
                    else:
                        # If transcription doesn't directly extend (e.g., correction or new segment),
                        # compare with the previous transcription to find differences or send the whole new one.
                        # For simplicity, if not a direct extension, we'll just send the current full transcription
                        # for now, and rely on the client to update/replace.
                        # A more robust solution would use difflib or word-level timestamps.
                        new_text = full_transcription # Send the whole updated transcription

                    if new_text:
                        print(f"[{client_id}][{source_type}] Live Transcription: {new_text}")
                        # Send as a 'live' or 'partial' update
                        await websocket.send_text(f"{source_type.capitalize()}LiveTranscription: {new_text}")
                        previous_full_transcription[source_type][client_id] = full_transcription

                    # After a period of silence or if full transcription changes significantly,
                    # we can also send a 'final' transcription and reset.
                    # This logic is more complex and usually involves VAD on the server too,
                    # or detecting natural pauses. For now, rely on `if new_text` for updates.

                    # Decide when to send suggestions based on natural utterance breaks or a final thought
                    # This could be linked to silence detection, or completion of a coherent sentence.
                    # For a simple demo, we'll continue sending suggestions after a certain amount of text
                    # or after a transcription has stabilized for a bit.
                    # As `make_suggestion` uses `conversation_history`, it implicitly works on cumulative text.
                    # We'll trigger it less frequently than live transcription updates.
                    # Trigger suggestion only when a significant new segment is transcribed and processed
                    if len(full_transcription) > 0 and len(full_transcription.split()) >= 5 and (
                        current_time - last_transcription_time[source_type][client_id] > TRANSCRIBE_STRIDE_SECONDS * 2
                        or new_text.endswith(('.', '?', '!', '\n'))
                    ):
                        suggestion = await make_suggestion(client_id, source_type, full_transcription)
                        await websocket.send_text(f"{source_type.capitalize()}Suggestion: {suggestion}")
                        # Reset previous_full_transcription after a "final" segment for suggestion
                        # to ensure the next segment starts fresh
                        # previous_full_transcription[source_type][client_id] = ""

                except Exception as e:
                    print(f"[{client_id}][{source_type}] Error during transcription or suggestion: {e}")
                    await websocket.send_text(f"{source_type.capitalize()}ServerError: Failed to process audio.")
                    import traceback
                    traceback.print_exc()

    except WebSocketDisconnect:
        print(f"[-] Client {client_id} disconnected from {source_type} audio.")
        audio_buffers[source_type].pop(client_id, None)
        last_transcription_time[source_type].pop(client_id, None)
        previous_full_transcription[source_type].pop(client_id, None)
        conversation_history[source_type].pop(client_id, None)
    except Exception as e:
        print(f"Error for client {client_id} on {source_type} audio: {e}")
        import traceback
        traceback.print_exc()
        # Clean up in case of other errors
        if client_id in audio_buffers[source_type]:
            audio_buffers[source_type].pop(client_id, None)
            last_transcription_time[source_type].pop(client_id, None)
            previous_full_transcription[source_type].pop(client_id, None)
            conversation_history[source_type].pop(client_id, None)
        try:
            await websocket.close()
        except RuntimeError:
            pass

# --- FastAPI WebSocket Endpoints ---
@app.websocket("/ws_mic")
async def websocket_mic_endpoint(websocket: WebSocket):
    await handle_audio_websocket(websocket, "mic")

@app.websocket("/ws_tab")
async def websocket_tab_endpoint(websocket: WebSocket):
    await handle_audio_websocket(websocket, "tab")

# --- HTML Content for Client ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>FastAPI Mic & Tab Audio WebSocket</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        button { padding: 10px 15px; margin: 5px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }

        .tab-buttons { display: flex; margin-bottom: 10px; }
        .tab-button {
            padding: 10px 15px;
            cursor: pointer;
            border: 1px solid #ccc;
            border-bottom: none;
            background-color: #f1f1f1;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            flex-grow: 1;
            text-align: center;
        }
        .tab-button.active {
            background-color: #e0e0e0;
            border-bottom: 1px solid #e0e0e0;
        }
        .tab-content {
            border: 1px solid #ccc;
            padding: 10px;
            min-height: 200px;
            background-color: #e0e0e0;
            border-radius: 5px;
            overflow-y: auto;
            max-height: 400px;
        }
        .message-item {
            padding: 8px;
            margin-bottom: 5px;
            background-color: #f9f9f9;
            border-radius: 5px;
            border-left: 3px solid #007bff;
            word-wrap: break-word;
        }
        .suggestion-item {
            padding: 8px;
            margin-bottom: 5px;
            background-color: #e6ffe6;
            border-radius: 5px;
            border-left: 3px solid #28a745;
            word-wrap: break-word;
        }
        .live-transcription-area {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            min-height: 20px; /* To prevent collapse when empty */
        }
    </style>
</head>
<body>
    <h1>FastAPI Mic & Tab Audio WebSocket</h1>
    <button id="startMicButton">Start Microphone</button>
    <button id="stopMicButton" disabled>Stop Microphone</button>
    <button id="startTabButton">Start Tab Audio</button>
    <button id="stopTabButton" disabled>Stop Tab Audio</button>

    <div class="tab-buttons">
        <button class="tab-button active" onclick="openTab(event, 'micTab')">Microphone Output</button>
        <button class="tab-button" onclick="openTab(event, 'tabTab')">Tab Audio Output</button>
    </div>

    <div id="micTab" class="tab-content">
        <h2>Microphone Live Transcriptions</h2>
        <div id="micLiveTranscription" class="live-transcription-area"></div>
        <h2>Microphone Final Transcriptions & Suggestions</h2>
        <ul id="micTranscriptionMessages"></ul>
        <ul id="micSuggestionMessages"></ul>
    </div>

    <div id="tabTab" class="tab-content" style="display:none;">
        <h2>Tab Audio Live Transcriptions</h2>
        <div id="tabLiveTranscription" class="live-transcription-area"></div>
        <h2>Tab Audio Final Transcriptions & Suggestions</h2>
        <ul id="tabTranscriptionMessages"></ul>
        <ul id="tabSuggestionMessages"></ul>
    </div>

    <script>
        const wsMic = new WebSocket("ws://localhost:8000/ws_mic");
        const wsTab = new WebSocket("ws://localhost:8000/ws_tab");

        const micLiveTranscription = document.getElementById('micLiveTranscription');
        const micTranscriptionMessages = document.getElementById('micTranscriptionMessages');
        const micSuggestionMessages = document.getElementById('micSuggestionMessages');

        const tabLiveTranscription = document.getElementById('tabLiveTranscription');
        const tabTranscriptionMessages = document.getElementById('tabTranscriptionMessages');
        const tabSuggestionMessages = document.getElementById('tabSuggestionMessages');

        const startMicButton = document.getElementById('startMicButton');
        const stopMicButton = document.getElementById('stopMicButton');
        const startTabButton = document.getElementById('startTabButton');
        const stopTabButton = document.getElementById('stopTabButton');

        let micStreamProcessor = null;
        let tabStreamProcessor = null;

        // --- WebSocket Handlers for Microphone ---
        wsMic.onopen = function(event) {
            console.log("Mic WebSocket connection opened:", event);
            addMessageToLog(micTranscriptionMessages, '<em>Connected to microphone service</em>', 'message-item');
            startMicButton.disabled = false;
        };
        wsMic.onmessage = function(event) {
            handleWsMessage(event.data, micLiveTranscription, micTranscriptionMessages, micSuggestionMessages, 'Mic');
        };
        wsMic.onclose = function(event) {
            console.log("Mic WebSocket connection closed:", event);
            addMessageToLog(micTranscriptionMessages, '<em>Disconnected from microphone service</em>', 'message-item');
            startMicButton.disabled = false;
            stopMicButton.disabled = true;
            stopAudioStream('mic');
        };
        wsMic.onerror = function(event) {
            console.error("Mic WebSocket error:", event);
            addMessageToLog(micTranscriptionMessages, '<li style="color: red;"><em>Mic WebSocket error!</em></li>', 'message-item');
            startMicButton.disabled = false;
            stopMicButton.disabled = true;
            stopAudioStream('mic');
        };

        // --- WebSocket Handlers for Tab Audio ---
        wsTab.onopen = function(event) {
            console.log("Tab WebSocket connection opened:", event);
            addMessageToLog(tabTranscriptionMessages, '<em>Connected to tab audio service</em>', 'message-item');
            startTabButton.disabled = false;
        };
        wsTab.onmessage = function(event) {
            handleWsMessage(event.data, tabLiveTranscription, tabTranscriptionMessages, tabSuggestionMessages, 'Tab');
        };
        wsTab.onclose = function(event) {
            console.log("Tab WebSocket connection closed:", event);
            addMessageToLog(tabTranscriptionMessages, '<em>Disconnected from tab audio service</em>', 'message-item');
            startTabButton.disabled = false;
            stopTabButton.disabled = true;
            stopAudioStream('tab');
        };
        wsTab.onerror = function(event) {
            console.error("Tab WebSocket error:", event);
            addMessageToLog(tabTranscriptionMessages, '<li style="color: red;"><em>Tab WebSocket error!</em></li>', 'message-item');
            startTabButton.disabled = false;
            stopTabButton.disabled = true;
            stopAudioStream('tab');
        };

        // --- Generic WebSocket Message Handler ---
        function handleWsMessage(message, liveEl, logEl, suggestionEl, sourceName) {
            if (message.startsWith(`${sourceName}LiveTranscription:`)) {
                liveEl.textContent = message.replace(`${sourceName}LiveTranscription: `, '');
            } else if (message.startsWith(`${sourceName}Transcription:`)) {
                // This would be for "final" transcriptions after an utterance or pause
                addMessageToLog(logEl, message.replace(`${sourceName}Transcription: `, ''), 'message-item');
                liveEl.textContent = ''; // Clear live transcription when a final one is logged
            } else if (message.startsWith(`${sourceName}Suggestion:`)) {
                addMessageToLog(suggestionEl, message.replace(`${sourceName}Suggestion: `, ''), 'suggestion-item');
            } else if (message.startsWith(`${sourceName}ServerError:`)) {
                addMessageToLog(logEl, `<li style="color: red;">${message.replace(`${sourceName}ServerError: `, '')}</li>`, 'message-item');
            } else {
                console.warn(`Unexpected message from ${sourceName} service:`, message);
            }
        }

        // --- Microphone Control ---
        startMicButton.onclick = async function() {
            startMicButton.disabled = true;
            stopMicButton.disabled = false;
            addMessageToLog(micTranscriptionMessages, '<em>Starting microphone...</em>', 'message-item');

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.audioWorklet.addModule('audio-processor.js');

                const microphone = audioContext.createMediaStreamSource(stream);
                micStreamProcessor = new AudioWorkletNode(audioContext, 'audio-data-processor');

                microphone.connect(micStreamProcessor);
                micStreamProcessor.connect(audioContext.destination);

                micStreamProcessor.port.onmessage = (event) => {
                    if (wsMic.readyState === WebSocket.OPEN) {
                        wsMic.send(event.data);
                    }
                };
                addMessageToLog(micTranscriptionMessages, '<em>Microphone started. Sending audio data.</em>', 'message-item');
                micStreamProcessor.stream = stream;

            } catch (err) {
                console.error('Error accessing microphone:', err);
                addMessageToLog(micTranscriptionMessages, '<li style="color: red;"><em>Error accessing microphone: ' + err.message + '</em></li>', 'message-item');
                startMicButton.disabled = false;
                stopMicButton.disabled = true;
            }
        };

        stopMicButton.onclick = function() {
            stopAudioStream('mic');
            startMicButton.disabled = false;
            stopMicButton.disabled = true;
            addMessageToLog(micTranscriptionMessages, '<em>Microphone stopped.</em>', 'message-item');
            micLiveTranscription.textContent = ''; // Clear live display on stop
        };

        // --- Tab Audio Control ---
        startTabButton.onclick = async function() {
            startTabButton.disabled = true;
            stopTabButton.disabled = false;
            addMessageToLog(tabTranscriptionMessages, '<em>Starting tab audio capture...</em>', 'message-item');

            try {
                const displayMediaOptions = {
                    video: true,
                    audio: true,
                };
                const stream = await navigator.mediaDevices.getDisplayMedia(displayMediaOptions);

                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.audioWorklet.addModule('audio-processor.js');

                const tabAudioSource = audioContext.createMediaStreamSource(stream);
                tabStreamProcessor = new AudioWorkletNode(audioContext, 'audio-data-processor');

                tabAudioSource.connect(tabStreamProcessor);
                tabStreamProcessor.connect(audioContext.destination);

                tabStreamProcessor.port.onmessage = (event) => {
                    if (wsTab.readyState === WebSocket.OPEN) {
                        wsTab.send(event.data);
                    }
                };
                addMessageToLog(tabTranscriptionMessages, '<em>Tab audio started. Sending audio data.</em>', 'message-item');
                tabStreamProcessor.stream = stream;

            } catch (err) {
                console.error('Error accessing tab audio:', err);
                addMessageToLog(tabTranscriptionMessages, '<li style="color: red;"><em>Error accessing tab audio: ' + err.message + '</em></li>', 'message-item');
                startTabButton.disabled = false;
                stopTabButton.disabled = true;
            }
        };

        stopTabButton.onclick = function() {
            stopAudioStream('tab');
            startTabButton.disabled = false;
            stopTabButton.disabled = true;
            addMessageToLog(tabTranscriptionMessages, '<em>Tab audio stopped.</em>', 'message-item');
            tabLiveTranscription.textContent = ''; // Clear live display on stop
        };

        // --- Helper Function to Stop Audio Streams ---
        function stopAudioStream(type) {
            let processor;
            if (type === 'mic' && micStreamProcessor) {
                processor = micStreamProcessor;
                micStreamProcessor = null;
            } else if (type === 'tab' && tabStreamProcessor) {
                processor = tabStreamProcessor;
                tabStreamProcessor = null;
            } else {
                return;
            }

            if (processor && processor.stream) {
                processor.stream.getTracks().forEach(track => track.stop());
            }
        }

        // --- Tab Management ---
        function openTab(evt, tabName) {
            const tabContents = document.getElementsByClassName("tab-content");
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].style.display = "none";
            }

            const tabButtons = document.getElementsByClassName("tab-button");
            for (let i = 0; i < tabButtons.length; i++) {
                tabButtons[i].className = tabButtons[i].className.replace(" active", "");
            }

            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }

        // Function to add messages to log lists (not live transcription area)
        function addMessageToLog(tabElement, msg, className = '') {
            const li = document.createElement('li');
            li.className = className;
            li.innerHTML = msg;
            tabElement.appendChild(li);
            tabElement.scrollTop = tabElement.scrollHeight;
        }

        document.addEventListener("DOMContentLoaded", () => {
            document.querySelector(".tab-button").click();
        });

    </script>
</body>
</html>
"""

# --- AudioWorklet Processor (JavaScript File) ---
# ... (audio_processor_js - no changes needed from the VAD version) ...
audio_processor_js = """
// audio-processor.js
class AudioDataProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 16000;
        this.buffer = [];
        this.lastUpdateTime = 0;
        this.sendInterval = 1000; // Client sends 1-second chunks

        this.inputSampleRate = 0;
        this.resampler = null;

        this.silenceThreshold = 0.005;
        this.minSilentChunks = 3;
        this.consecutiveSilentChunks = 0;
        this.wasSendingAudio = false;
    }

    initResampler(inputSampleRate) {
        if (inputSampleRate === this.sampleRate) {
            this.resampler = null;
        } else {
            this.resampler = (inputBuffer) => {
                const outputBuffer = new Float32Array(Math.ceil(inputBuffer.length * (this.sampleRate / inputSampleRate)));
                const ratio = inputSampleRate / this.sampleRate;
                for (let i = 0; i < outputBuffer.length; i++) {
                    const index = i * ratio;
                    const lower = Math.floor(index);
                    const upper = Math.ceil(index);
                    const weight = index - lower;

                    if (upper < inputBuffer.length) {
                        outputBuffer[i] = inputBuffer[lower] * (1 - weight) + inputBuffer[upper] * weight;
                    } else {
                        outputBuffer[i] = inputBuffer[lower];
                    }
                }
                return outputBuffer;
            };
        }
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length === 0) {
            return true;
        }

        const inputChannelData = input[0];

        if (this.inputSampleRate === 0) {
            this.inputSampleRate = sampleRate;
            this.initResampler(this.inputSampleRate);
        }

        let processedData = inputChannelData;
        if (this.resampler) {
            processedData = this.resampler(inputChannelData);
        }

        const rms = Math.sqrt(processedData.reduce((sum, value) => sum + value * value, 0) / processedData.length);

        const isVoice = rms > this.silenceThreshold;

        if (isVoice) {
            this.consecutiveSilentChunks = 0;
            this.buffer.push(processedData);
            this.wasSendingAudio = true;
        } else {
            this.consecutiveSilentChunks++;
            if (this.wasSendingAudio && this.buffer.length > 0) {
                const totalLength = this.buffer.reduce((acc, val) => acc + val.length, 0);
                const combinedBuffer = new Float32Array(totalLength);
                let offset = 0;
                for (const array of this.buffer) {
                    combinedBuffer.set(array, offset);
                    offset += array.length;
                }
                this.port.postMessage(combinedBuffer.buffer, [combinedBuffer.buffer]);
                this.buffer = [];
                this.wasSendingAudio = false;
            }

            if (this.consecutiveSilentChunks > this.minSilentChunks) {
                this.buffer = [];
                this.lastUpdateTime = Date.now();
                return true;
            } else {
                this.buffer.push(processedData);
            }
        }

        const currentTime = Date.now();
        if (this.buffer.length > 0 && currentTime - this.lastUpdateTime > this.sendInterval) {
            const totalLength = this.buffer.reduce((acc, val) => acc + val.length, 0);
            const combinedBuffer = new Float32Array(totalLength);
            let offset = 0;
            for (const array of this.buffer) {
                combinedBuffer.set(array, offset);
                offset += array.length;
            }

            this.port.postMessage(combinedBuffer.buffer, [combinedBuffer.buffer]);
            this.buffer = [];
            this.lastUpdateTime = currentTime;
            this.wasSendingAudio = false;
        }

        return true;
    }
}

registerProcessor('audio-data-processor', AudioDataProcessor);
"""

# --- FastAPI Routes ---
@app.get("/")
async def get_html():
    return HTMLResponse(html_content)

@app.get("/audio-processor.js")
async def get_audio_processor_js():
    return HTMLResponse(audio_processor_js, media_type="application/javascript")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
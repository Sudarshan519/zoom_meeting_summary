import asyncio
import io
import os
import time
import warnings
import wave

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv
import markdown
import uvicorn
import whisper
from openai import AsyncOpenAI 
# --- Initialize API Client and Load Model ---
load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
model = whisper.load_model("base")
# ---------------------------------------------
suggestion_locks = {}
app = FastAPI()
def is_suggestion_pending(client_id, source_type):
    return suggestion_locks.get((client_id, source_type), False)
# --- Audio Processing Constants ---
sample_rate = 16000
channels = 1
bytes_per_sample = 4 # float32 uses 4 bytes per sample
buffer_threshold_seconds = 2 # Process audio in 5-second chunks
buffer_threshold = sample_rate * bytes_per_sample * buffer_threshold_seconds

# --- Buffers and Histories (per client, per source) ---
# Each client will have separate buffers/histories for mic and tab audio
audio_buffers = {
    "mic": {}, # {client_id: BytesIO}
    "tab": {}  # {client_id: BytesIO}
}
conversation_history = {
    "mic": {}, # {client_id: [utterances]}
    "tab": {}  # {client_id: [utterances]}
}
# -----------------------------------------------

# --- OpenAI Suggestion Function ---
async def make_suggestion(client_id: str, source_type: str, text: str):
    print(f"🤖 Sending to GPT-4o for analysis ({source_type}) for {client_id}...")
    prompt = "You're assisting a user during a meeting. Format conversation and also provide possible answers to questions. Format the answer as a markdown list or paragraph.Conversation: [Question] [Answer] {Answer}"

    if client_id not in conversation_history[source_type]:
        conversation_history[source_type][client_id] = []
    conversation_history[source_type][client_id].append(text)

    context_text = "\n\n".join(conversation_history[source_type][client_id][-5:])

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context_text}
    ]

    try:
        completion =await  client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response_content = completion.choices[0].message.content
        print(f"GPT-4o response ({source_type}) for {client_id}: {response_content}")
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API ({source_type}) for {client_id}: {e}")
        return "Error generating suggestion."
# --- Import run_in_threadpool ---
from fastapi.concurrency import run_in_threadpool
import asyncio
async def send_suggestion_later(client_id, source_type, transcription, websocket):
    key = (client_id, source_type)
    
    # Skip if already processing a suggestion for this client/source
    if suggestion_locks.get(key):
        print(f"Skipping suggestion for {key}, already in progress.")
        return

    suggestion_locks[key] = True
    try:
        suggestion = await make_suggestion(client_id, source_type, transcription)
        await websocket.send_text(f"{source_type.capitalize()}Suggestion: {suggestion}")
    except Exception as e:
        print(f"Failed to generate/send suggestion for {key}: {e}")
    finally:
        suggestion_locks[key] = False  # Clear lock
# --- WebSocket Processing Handler (reusable for both mic and tab) ---
async def handle_audio_websocket(websocket: WebSocket, source_type: str):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"[+] Client {client_id} connected for {source_type} audio.")

    audio_buffers[source_type][client_id] = io.BytesIO()
    conversation_history[source_type][client_id] = []

    try:
        while True:
            audio_data_bytes = await websocket.receive_bytes()
            audio_buffers[source_type][client_id].write(audio_data_bytes)

            if audio_buffers[source_type][client_id].tell() >= buffer_threshold:
                print(f"[{client_id}][{source_type}] Processing {audio_buffers[source_type][client_id].tell()} bytes of audio...")

                audio_buffers[source_type][client_id].seek(0)
                audio_np = np.frombuffer(audio_buffers[source_type][client_id].read(), dtype=np.float32)
                # --- ADD THIS CHECK ---
                is_silent = np.all(audio_np == 0) # Check if all samples are zero
                if is_silent and audio_np.size > 0:
                    print(f"[{client_id}][{source_type}] Received SILENT audio chunk ({audio_np.size} samples).")
                    # Optional: You could choose NOT to transcribe silent chunks here
                    # audio_buffers[source_type][client_id].seek(0); audio_buffers[source_type][client_id].truncate(0)
                    # continue # Skip to next loop iteration
                # ---------------------

                try:
                    # --- CRUCIAL CHANGE: Run transcribe in a thread pool ---
                    transcription_result = await run_in_threadpool(
                        model.transcribe, audio_np, fp16=False
                    )
                    # -----------------------------------------------------

                    # transcription_result = model.transcribe(audio_np, fp16=False)
                    transcription = transcription_result['text'].strip()
                    conversation_history[source_type][client_id].append(transcription)
                    print(f"[{client_id}][{source_type}] Transcription: {transcription}")

                    if transcription:
                        await websocket.send_text(f"{source_type.capitalize()}Transcription: {transcription}")
                        # asyncio.create_task(send_suggestion_later(client_id, source_type, transcription, websocket))
                        # suggestion = await make_suggestion(client_id, source_type, transcription)
                        # await websocket.send_text(f"{source_type.capitalize()}Suggestion: {suggestion}")

                except Exception as e:
                    print(f"[{client_id}][{source_type}] Error during transcription or suggestion: {e}")
                    await websocket.send_text(f"{source_type.capitalize()}ServerError: Failed to process audio.")
                    import traceback
                    traceback.print_exc()

                audio_buffers[source_type][client_id].seek(0)
                audio_buffers[source_type][client_id].truncate(0)

    except WebSocketDisconnect:
        print(f"[-] Client {client_id} disconnected from {source_type} audio.")
        audio_buffers[source_type].pop(client_id, None)
        conversation_history[source_type].pop(client_id, None)
    except Exception as e:
        print(f"Error for client {client_id} on {source_type} audio: {e}")
        import traceback
        traceback.print_exc()
        if client_id in audio_buffers[source_type]:
            audio_buffers[source_type].pop(client_id, None)
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
        <h2>Microphone Transcriptions & Suggestions</h2>
        <ul id="micTranscriptionMessages"></ul>
        <ul id="micSuggestionMessages"></ul>
    </div>

    <div id="tabTab" class="tab-content" style="display:none;">
        <h2>Tab Audio Transcriptions & Suggestions</h2>
        <ul id="tabTranscriptionMessages"></ul>
        <ul id="tabSuggestionMessages"></ul>
    </div>

    <script>

            // Silence detection
        function isReallySilent(audioBuffer, threshold = 0.01) {
            let energy = 0;
            for (let i = 0; i < audioBuffer.length; i++) {
                energy += audioBuffer[i] * audioBuffer[i];
            }
            return Math.sqrt(energy / audioBuffer.length) < threshold;
        }
        // Separate WebSockets for mic and tab audio
        const wsMic = new WebSocket("ws://localhost:8000/ws_mic");
        const wsTab = new WebSocket("ws://localhost:8000/ws_tab");

        // UI elements
        const micTranscriptionMessages = document.getElementById('micTranscriptionMessages');
        const micSuggestionMessages = document.getElementById('micSuggestionMessages');
        const tabTranscriptionMessages = document.getElementById('tabTranscriptionMessages');
        const tabSuggestionMessages = document.getElementById('tabSuggestionMessages');

        const startMicButton = document.getElementById('startMicButton');
        const stopMicButton = document.getElementById('stopMicButton');
        const startTabButton = document.getElementById('startTabButton');
        const stopTabButton = document.getElementById('stopTabButton');

        let micStreamProcessor = null; // For microphone AudioWorklet
        let tabStreamProcessor = null; // For tab AudioWorklet

        // --- WebSocket Handlers for Microphone ---
        wsMic.onopen = function(event) {
            console.log("Mic WebSocket connection opened:", event);
            addMessageToTab(micTranscriptionMessages, '<em>Connected to microphone service</em>', 'message-item');
            startMicButton.disabled = false;
        };
        wsMic.onmessage = function(event) {
            handleWsMessage(event.data, micTranscriptionMessages, micSuggestionMessages, 'Mic');
        };
        wsMic.onclose = function(event) {
            console.log("Mic WebSocket connection closed:", event);
            addMessageToTab(micTranscriptionMessages, '<em>Disconnected from microphone service</em>', 'message-item');
            startMicButton.disabled = false;
            stopMicButton.disabled = true;
            stopAudioStream('mic');
        };
        wsMic.onerror = function(event) {
            console.error("Mic WebSocket error:", event);
            addMessageToTab(micTranscriptionMessages, '<li style="color: red;"><em>Mic WebSocket error!</em></li>', 'message-item');
            startMicButton.disabled = false;
            stopMicButton.disabled = true;
            stopAudioStream('mic');
        };

        // --- WebSocket Handlers for Tab Audio ---
        wsTab.onopen = function(event) {
            console.log("Tab WebSocket connection opened:", event);
            addMessageToTab(tabTranscriptionMessages, '<em>Connected to tab audio service</em>', 'message-item');
            startTabButton.disabled = false;
        };
        wsTab.onmessage = function(event) {
            handleWsMessage(event.data, tabTranscriptionMessages, tabSuggestionMessages, 'Tab');
        };
        wsTab.onclose = function(event) {
            console.log("Tab WebSocket connection closed:", event);
            addMessageToTab(tabTranscriptionMessages, '<em>Disconnected from tab audio service</em>', 'message-item');
            startTabButton.disabled = false;
            stopTabButton.disabled = true;
            stopAudioStream('tab');
        };
        wsTab.onerror = function(event) {
            console.error("Tab WebSocket error:", event);
            addMessageToTab(tabTranscriptionMessages, '<li style="color: red;"><em>Tab WebSocket error!</em></li>', 'message-item');
            startTabButton.disabled = false;
            stopTabButton.disabled = true;
            stopAudioStream('tab');
        };

        // --- Generic WebSocket Message Handler ---
        function handleWsMessage(message, transcriptionEl, suggestionEl, sourceName) {
            if (message.startsWith(`${sourceName}Transcription:`)) {
                addMessageToTab(transcriptionEl, message.replace(`${sourceName}Transcription: `, ''), 'message-item');
            } else if (message.startsWith(`${sourceName}Suggestion:`)) {
                addMessageToTab(suggestionEl, message.replace(`${sourceName}Suggestion: `, ''), 'suggestion-item');
            } else if (message.startsWith(`${sourceName}ServerError:`)) {
              //  addMessageToTab(transcriptionEl, `<li style="color: red;">${message.replace(`${sourceName}ServerError: `, '')}</li>`, 'message-item');
            } else {
                console.warn(`Unexpected message from ${sourceName} service:`, message);
            }
        }

        // --- Microphone Control ---
        startMicButton.onclick = async function() {
            startMicButton.disabled = true;
            stopMicButton.disabled = false;
            addMessageToTab(micTranscriptionMessages, '<em>Starting microphone...</em>', 'message-item');

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.audioWorklet.addModule('audio-processor.js');

                const microphone = audioContext.createMediaStreamSource(stream);
                micStreamProcessor = new AudioWorkletNode(audioContext, 'audio-data-processor'); // Store reference

                microphone.connect(micStreamProcessor);
                micStreamProcessor.connect(audioContext.destination);

                micStreamProcessor.port.onmessage = (event) => {
                    if (wsMic.readyState === WebSocket.OPEN) {
                    const audioBuffer = new Float32Array(event.data); // Convert ArrayBuffer to Float32Array
                 // if(  !isReallySilent(audioBuffer))
                        wsMic.send(event.data);
                    }
                };
                addMessageToTab(micTranscriptionMessages, '<em>Microphone started. Sending audio data.</em>', 'message-item');
                micStreamProcessor.stream = stream; // Store stream to stop tracks later

            } catch (err) {
                console.error('Error accessing microphone:', err);
                addMessageToTab(micTranscriptionMessages, '<li style="color: red;"><em>Error accessing microphone: ' + err.message + '</em></li>', 'message-item');
                startMicButton.disabled = false;
                stopMicButton.disabled = true;
            }
        };

        stopMicButton.onclick = function() {
            stopAudioStream('mic');
            startMicButton.disabled = false;
            stopMicButton.disabled = true;
            addMessageToTab(micTranscriptionMessages, '<em>Microphone stopped.</em>', 'message-item');
        };

        // --- Tab Audio Control ---
        startTabButton.onclick = async function() {
            startTabButton.disabled = true;
            stopTabButton.disabled = false;
            addMessageToTab(tabTranscriptionMessages, '<em>Starting tab audio capture...</em>', 'message-item');

            try {
                // `displayMediaOptions` can be used to specifically ask for browser tab
                const displayMediaOptions = {
                    video: true, // Don't need video
                    audio: {
                        latency: 0, // Minimize latency
                        sampleRate: 16000, // Request specific sample rate if possible
                        channelCount: 1, // Request mono if possible
                    },
                    // This is for browser-specific options, not standardized for all browsers.
                    // Check browser compatibility for 'preferCurrentTab'.
                    // mediaSource: 'browser' // Example for Chrome (non-standard)
                };
                // `getDisplayMedia` shows a dialog to the user to pick screen/window/tab
              //  const stream = await navigator.mediaDevices.getDisplayMedia(displayMediaOptions);
const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true // Only works for Chrome tab sharing
    });
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.audioWorklet.addModule('audio-processor.js');

                const tabAudioSource = audioContext.createMediaStreamSource(stream);
                tabStreamProcessor = new AudioWorkletNode(audioContext, 'audio-data-processor'); // Store reference

                tabAudioSource.connect(tabStreamProcessor);
                tabStreamProcessor.connect(audioContext.destination);

                tabStreamProcessor.port.onmessage = (event) => {
                    if (wsTab.readyState === WebSocket.OPEN) {
                         const audioBuffer = new Float32Array(event.data); // Convert ArrayBuffer to Float32Array
                  if(  !isReallySilent(audioBuffer))
                      wsTab.send(event.data);
                
                       
                    }
                };
                addMessageToTab(tabTranscriptionMessages, '<em>Tab audio started. Sending audio data.</em>', 'message-item');
                tabStreamProcessor.stream = stream; // Store stream to stop tracks later

            } catch (err) {
                console.error('Error accessing tab audio:', err);
                addMessageToTab(tabTranscriptionMessages, '<li style="color: red;"><em>Error accessing tab audio: ' + err.message + '</em></li>', 'message-item');
                startTabButton.disabled = false;
                stopTabButton.disabled = true;
            }
        };

        stopTabButton.onclick = function() {
            stopAudioStream('tab');
            startTabButton.disabled = false;
            stopTabButton.disabled = true;
            addMessageToTab(tabTranscriptionMessages, '<em>Tab audio stopped.</em>', 'message-item');
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
                processor.stream.getTracks().forEach(track => track.stop()); // Stop all tracks in the stream
            }
            if (processor && processor.context) { // Try to close AudioContext, though it might be shared
                // processor.context.close(); // Careful: If AudioContext is shared, don't close it this way
            }
            // Disconnect nodes to stop processing
            if (processor && processor.disconnect) {
                 // Disconnect from input and output to stop processing
                 // This requires keeping references to source nodes too, for a full clean up.
                 // For simplicity in demo, stopping tracks is the most effective.
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

        // Function to add messages to specific tabs
        function addMessageToTab(tabElement, msg, className = '') {
            const li = document.createElement('li');
           li.className = className;
            li.innerHTML = msg;
            tabElement.prepend(li);
           tabElement.scrollTop = tabElement.scrollHeight;
         //  tabElement.innerHTML = `${msg}</li>`;
        }

        // Open the default tab on page load
        document.addEventListener("DOMContentLoaded", () => {
            document.querySelector(".tab-button").click();
        });

    </script>
</body>
</html>
"""


# --- AudioWorklet Processor: `audio-processor.js` ---
# This script runs in a separate thread in the browser to handle audio capture and pre-processing.
audio_processor_js = """
// audio-processor.js
// This AudioWorkletProcessor handles raw audio data from the microphone/tab,
// performs resampling, basic Voice Activity Detection (VAD), and buffers
// audio to send to the server at regular intervals.
class AudioDataProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 16000; // Target sample rate for Whisper
        this.buffer = []; // Buffer to accumulate audio data
        this.lastUpdateTime = 0;
        this.sendInterval = 3000; // Send audio chunks to server every 1000ms (1 second)

        this.inputSampleRate = 0; // Actual sample rate of the input audio stream
        this.resampler = null; // Function to resample audio if needed

        this.silenceThreshold = 0.006; // RMS threshold for Voice Activity Detection (VAD)
        this.minSilentChunks = 1;    // Number of consecutive silent chunks before stopping sending
        this.consecutiveSilentChunks = 2;
        this.wasSendingAudio = false; // Flag to track if we were just sending voice

        // Utility function to determine if an audio buffer is "really silent"
        // This is a stricter check used just before sending to WebSocket.
        this.isReallySilent = (audioBuffer, threshold = 0.001) => {
            if (!audioBuffer || audioBuffer.length === 0) return true;
            let energy = 0;
            for (let i = 0; i < audioBuffer.length; i++) {
                energy += audioBuffer[i] * audioBuffer[i];
            }
            // Calculate RMS (Root Mean Square)
            return Math.sqrt(energy / audioBuffer.length) < threshold;
        };
    }

    // Initializes the resampler function if the input sample rate doesn't match the target
    initResampler(inputSampleRate) {
        if (inputSampleRate === this.sampleRate) {
            this.resampler = null; // No resampling needed
        } else {
            // Simple linear interpolation resampler
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

    // Main processing loop for incoming audio data
    process(inputs, outputs, parameters) {
        const input = inputs[0]; // Get the first input stream
        if (input.length === 0) {
            return true; // No input data, continue processing
        }

        const inputChannelData = input[0]; // Get the audio data from the first channel

        // Initialize resampler if not already done
        if (this.inputSampleRate === 0) {
            this.inputSampleRate = sampleRate; // `sampleRate` is globally available in AudioWorklet
            this.initResampler(this.inputSampleRate);
        }

        let processedData = inputChannelData;
        if (this.resampler) {
            processedData = this.resampler(inputChannelData); // Resample if necessary
        }

        // Calculate RMS for VAD
        const rms = Math.sqrt(processedData.reduce((sum, value) => sum + value * value, 0) / processedData.length);
        const isVoice = rms > this.silenceThreshold; // Determine if current chunk contains voice

        if (isVoice) {
            this.consecutiveSilentChunks = 5; // Reset silent counter
            this.buffer.push(processedData); // Buffer the voice data
            this.wasSendingAudio = true; // Mark as actively sending voice
        } else {
            this.consecutiveSilentChunks++; // Increment silent counter
            if (this.wasSendingAudio && this.buffer.length > 0) {
                // If we were just sending voice and now detected silence,
                // send the remaining buffered audio to capture the tail end of speech.
                const totalLength = this.buffer.reduce((acc, val) => acc + val.length, 0);
                const combinedBuffer = new Float32Array(totalLength);
                let offset = 0;
                for (const array of this.buffer) {
                    combinedBuffer.set(array, offset);
                    offset += array.length;
                }
                this.port.postMessage(combinedBuffer.buffer, [combinedBuffer.buffer]); // Send the ArrayBuffer
                this.buffer = []; // Clear buffer after sending
                this.wasSendingAudio = false; // Reset flag
            }

            if (this.consecutiveSilentChunks > this.minSilentChunks) {
                // If prolonged silence, clear buffer and reset time to stop sending
                // This prevents sending empty data during long pauses.
                this.buffer = [];
                this.lastUpdateTime = Date.now();
                return true; // No audio to send in this case
            } else {
                // If not prolonged silence (i.e., a brief pause within an utterance),
                // continue to buffer silent chunks to maintain context.
                this.buffer.push(processedData);
            }
        }

        const currentTime = Date.now();
        // If buffer has accumulated enough data OR enough time has passed, send the buffered data.
        if (this.buffer.length > 0 && currentTime - this.lastUpdateTime > this.sendInterval) {
            const totalLength = this.buffer.reduce((acc, val) => acc + val.length, 0);
            const combinedBuffer = new Float32Array(totalLength);
            let offset = 0;
            for (const array of this.buffer) {
                combinedBuffer.set(array, offset);
                offset += array.length;
            }

            this.port.postMessage(combinedBuffer.buffer, [combinedBuffer.buffer]); // Send the ArrayBuffer
            this.buffer = []; // Clear buffer after sending
            this.lastUpdateTime = currentTime;
            this.wasSendingAudio = false; // Reset after sending a chunk
        }

        return true; // Keep the processor active
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
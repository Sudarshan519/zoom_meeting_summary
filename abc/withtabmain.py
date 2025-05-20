from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import io
import wave
import numpy as np
import os
import time
import warnings
from openai import OpenAI
from dotenv import load_dotenv
import markdown

# --- Initialize API Client and Load Model ---
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
import whisper
model = whisper.load_model("small")
# ---------------------------------------------

app = FastAPI()

# --- Audio Processing Constants and Buffers ---
audio_buffers = {}
sample_rate = 16000
channels = 1
bytes_per_sample = 4
buffer_threshold_seconds = 5
buffer_threshold = sample_rate * bytes_per_sample * buffer_threshold_seconds

conversation_history = {}
# -----------------------------------------------

# --- OpenAI Suggestion Function ---
async def make_suggestion(client_id: str, text: str):
    print(f"🤖 Sending to GPT-4o for analysis for {client_id}...")
    prompt = "You're assisting a user during a meeting. Summarize the conversation and also provide possible answers to questions. Format the answer as a markdown list or paragraph.Conversation: [Question] [Answer] {Answer}"

    if client_id not in conversation_history:
        conversation_history[client_id] = []
    conversation_history[client_id].append(text)

    context_text = "\n\n".join(conversation_history[client_id][-5:])

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
        print(f"GPT-4o response for {client_id}: {response_content}")
        return response_content
    except Exception as e:
        print(f"Error calling OpenAI API for {client_id}: {e}")
        return "Error generating suggestion."

# --- HTML Content for Client ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>FastAPI Raw Audio WebSocket</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        button { padding: 10px 15px; margin: 5px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }

        /* Tab styles */
        .tab-buttons { display: flex; margin-bottom: 10px; }
        .tab-button {
            padding: 10px 15px;
            cursor: pointer;
            border: 1px solid #ccc;
            border-bottom: none;
            background-color: #f1f1f1;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            flex-grow: 1; /* Make buttons fill available space */
            text-align: center;
        }
        .tab-button.active {
            background-color: #e0e0e0;
            border-bottom: 1px solid #e0e0e0; /* Matches active tab content background */
        }
        .tab-content {
            border: 1px solid #ccc;
            padding: 10px;
            min-height: 200px;
            background-color: #e0e0e0; /* Slightly different background for content */
            border-radius: 5px;
            overflow-y: auto; /* Enable scrolling for long content */
            max-height: 400px; /* Limit height for scrolling */
        }
        .message-item {
            padding: 8px;
            margin-bottom: 5px;
            background-color: #f9f9f9; /* Lighter background for individual messages */
            border-radius: 5px;
            border-left: 3px solid #007bff; /* Accent border */
            word-wrap: break-word; /* Ensure long words wrap */
        }
        .suggestion-item {
            padding: 8px;
            margin-bottom: 5px;
            background-color: #e6ffe6; /* Light green for suggestions */
            border-radius: 5px;
            border-left: 3px solid #28a745; /* Accent border */
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <h1>FastAPI Raw Audio WebSocket</h1>
    <button id="startButton">Start Microphone</button>
    <button id="stopButton" disabled>Stop Microphone</button>

    <div class="tab-buttons">
        <button class="tab-button active" onclick="openTab(event, 'transcriptionTab')">Transcription</button>
        <button class="tab-button" onclick="openTab(event, 'suggestionTab')">Suggestions</button>
    </div>

    <div id="transcriptionTab" class="tab-content">
        <ul id="transcriptionMessages"></ul>
    </div>

    <div id="suggestionTab" class="tab-content" style="display:none;">
        <ul id="suggestionMessages"></ul>
    </div>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const transcriptionMessages = document.getElementById('transcriptionMessages');
        const suggestionMessages = document.getElementById('suggestionMessages');
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');

        let mediaRecorder;
        let audioChunks = [];
        let intervalId;
        const sendInterval = 1000;

        // --- WebSocket Handlers ---
        ws.onopen = function(event) {
            console.log("WebSocket connection opened:", event);
            addMessageToTab(transcriptionMessages, '<em>Connected to server</em>', 'message-item');
            startButton.disabled = false;
        };

        ws.onmessage = function(event) {
            var message = event.data;
            console.log("Received message:", message);
            // Distinguish between transcription and suggestion messages
            if (message.startsWith("Transcription:")) {
                addMessageToTab(transcriptionMessages, message.replace("Transcription: ", ""), 'message-item');
            } else if (message.startsWith("Suggestion:")) {
                // Use innerHTML to render markdown from suggestions
                addMessageToTab(suggestionMessages, message.replace("Suggestion: ", ""), 'suggestion-item');
            } else {
                addMessageToTab(transcriptionMessages, 'Server: ' + message, 'message-item'); // Fallback for other messages
            }
        };

        ws.onclose = function(event) {
            console.log("WebSocket connection closed:", event);
            addMessageToTab(transcriptionMessages, '<em>Disconnected from server</em>', 'message-item');
            startButton.disabled = false;
            stopButton.disabled = true;
            stopRecording();
        };

        ws.onerror = function(event) {
            console.error("WebSocket error:", event);
            addMessageToTab(transcriptionMessages, '<li style="color: red;"><em>WebSocket error!</em></li>', 'message-item');
            startButton.disabled = false;
            stopButton.disabled = true;
            stopRecording();
        };

        // --- Microphone Control ---
        startButton.onclick = async function() {
            startButton.disabled = true;
            stopButton.disabled = false;
            addMessageToTab(transcriptionMessages, '<em>Starting microphone...</em>', 'message-item');

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.audioWorklet.addModule('audio-processor.js');

                const microphone = audioContext.createMediaStreamSource(stream);
                const audioProcessor = new AudioWorkletNode(audioContext, 'audio-data-processor');

                microphone.connect(audioProcessor);
                audioProcessor.connect(audioContext.destination);

                audioProcessor.port.onmessage = (event) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data);
                    }
                };
                addMessageToTab(transcriptionMessages, '<em>Microphone started. Sending audio data.</em>', 'message-item');

            } catch (err) {
                console.error('Error accessing microphone:', err);
                addMessageToTab(transcriptionMessages, '<li style="color: red;"><em>Error accessing microphone: ' + err.message + '</em></li>', 'message-item');
                startButton.disabled = false;
                stopButton.disabled = true;
            }
        };

        stopButton.onclick = function() {
            stopRecording();
            startButton.disabled = false;
            stopButton.disabled = true;
            addMessageToTab(transcriptionMessages, '<em>Microphone stopped.</em>', 'message-item');
        };

        function stopRecording() {
            // Logic to stop microphone input (e.g., stopping MediaRecorder or audio context nodes)
            // For the AudioWorklet example, stopping the stream source might be sufficient
            // Or disconnect the AudioWorkletNode if more explicit cleanup is needed
            // For simplicity, this example doesn't fully stop the AudioWorklet graph gracefully.
            // A production app would manage `stream.getTracks().forEach(track => track.stop());`
            // and `audioContext.close();`
        }

        // --- Tab Management ---
        function openTab(evt, tabName) {
            // Get all elements with class="tab-content" and hide them
            const tabContents = document.getElementsByClassName("tab-content");
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].style.display = "none";
            }

            // Get all elements with class="tab-button" and remove the "active" class
            const tabButtons = document.getElementsByClassName("tab-button");
            for (let i = 0; i < tabButtons.length; i++) {
                tabButtons[i].className = tabButtons[i].className.replace(" active", "");
            }

            // Show the current tab, and add an "active" class to the button that opened the tab
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }

        // Function to add messages to specific tabs
        function addMessageToTab(tabElement, msg, className = '') {
            const li = document.createElement('li');
            li.className = className; // Apply class for styling
            li.innerHTML = msg; // Use innerHTML to allow Markdown rendering in suggestions
            tabElement.appendChild(li);
            tabElement.scrollTop = tabElement.scrollHeight; // Auto-scroll
        }

        // Open the default tab on page load
        document.addEventListener("DOMContentLoaded", () => {
            document.querySelector(".tab-button").click(); // Click the first tab button
        });

    </script>
</body>
</html>
"""

# --- AudioWorklet Processor (JavaScript File) ---
# This code is served by FastAPI as 'audio-processor.js'.
audio_processor_js = """
// audio-processor.js (for AudioWorklet)
class AudioDataProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 16000; // Target sample rate
        this.buffer = []; // To accumulate samples
        this.lastUpdateTime = 0;
        this.sendInterval = 1000; // Send data every 1 second (1000ms)

        this.inputSampleRate = 0; // Will be set on first process call or via port message
        this.resampler = null;
    }

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

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length === 0) {
            return true;
        }

        const inputChannelData = input[0];

        // Lazily initialize resampler if inputSampleRate changes or is first determined
        if (this.inputSampleRate === 0) { // First process call, get actual sample rate
            // In AudioWorklet, `sampleRate` property is the context's sample rate
            this.inputSampleRate = this.sampleRate; // This `sampleRate` is the AudioContext's sampleRate
            this.initResampler(this.inputSampleRate);
        }


        let processedData = inputChannelData;
        if (this.resampler) {
            processedData = this.resampler(inputChannelData);
        }

        this.buffer.push(processedData);

        const currentTime = Date.now();
        if (currentTime - this.lastUpdateTime > this.sendInterval) {
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"[+] Client {client_id} connected via WebSocket.")

    audio_buffers[client_id] = io.BytesIO()
    conversation_history[client_id] = []

    try:
        while True:
            audio_data_bytes = await websocket.receive_bytes()
            audio_buffers[client_id].write(audio_data_bytes)

            if audio_buffers[client_id].tell() >= buffer_threshold:
                print(f"[{client_id}] Processing {audio_buffers[client_id].tell()} bytes of audio...")

                audio_buffers[client_id].seek(0)
                audio_np = np.frombuffer(audio_buffers[client_id].read(), dtype=np.float32)

                try:
                    transcription_result = model.transcribe(audio_np, fp16=False)
                    transcription = transcription_result['text'].strip()
                    print(f"[{client_id}] Transcription: {transcription}")

                    if transcription:
                        # Prefix messages to help client distinguish
                        await websocket.send_text(f"Transcription: {transcription}")

                        suggestion = await make_suggestion(client_id, transcription)
                        # Prefix messages to help client distinguish
                        await websocket.send_text(f"Suggestion: {suggestion}")

                except Exception as e:
                    print(f"[{client_id}] Error during transcription or suggestion: {e}")
                    await websocket.send_text("Server Error: Failed to process audio.")
                    import traceback
                    traceback.print_exc()

                audio_buffers[client_id].seek(0)
                audio_buffers[client_id].truncate(0)

    except WebSocketDisconnect:
        print(f"[-] Client {client_id} disconnected.")
        audio_buffers.pop(client_id, None)
        conversation_history.pop(client_id, None)
    except Exception as e:
        print(f"Error for client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        if client_id in audio_buffers:
            audio_buffers.pop(client_id, None)
            conversation_history.pop(client_id, None)
        try:
            await websocket.close()
        except RuntimeError:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
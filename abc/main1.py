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
import markdown # Assuming you still want to use markdown for suggestions

# --- Initialize API Client and Load Model ---
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
# Assuming whisper is installed, if not, `pip install "git+https://github.com/openai/whisper.git"`
import whisper
model = whisper.load_model("small")
# ---------------------------------------------

app = FastAPI()

# --- Audio Processing Constants and Buffers ---
audio_buffers = {} # Dictionary to store BytesIO buffers for each connected client
sample_rate = 16000 # Typical sample rate for speech (e.g., Whisper's default)
channels = 1        # Mono audio
bytes_per_sample = 4 # float32 uses 4 bytes per sample
buffer_threshold_seconds = 5 # Process audio in 5-second chunks
buffer_threshold = sample_rate * bytes_per_sample * buffer_threshold_seconds

conversation_history = {} # Store conversation history per client for context
# -----------------------------------------------
def makeSuggestion(text):
    print("🤖 Sending to GPT-4 for analysis...")
    # You are an expert in communication analysis. Analyze this short transcript:
    # - Emotional tone
    prompt ="You're assisting a user during a meeting. Summarize the conversation and also provide the possible answers to question.Conversation [Question] [Answer] {Answer}"# f"""You are an bot helping user on meetin with context.Separate conversation and also suggest feasible solutions to help host for feasible solutions.Suggest latest answer at top.Respond like as if you are the host of meeting.Respond only in english"""
    conversation_text = "\n\n"  # Join the conversation list into a single string with line breaks
    conversation_text += f"\n\n{text}"
    # print(conversation_text)
    messages = [
        {"role": "system", "content":prompt},# "You are a coach helping actors improve their responses during interviews or auditions."},
        {"role": "user", "content": conversation_text}
    ]

    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=messages,
    #     temperature=0.7,
    #     max_tokens=500
    # )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
        #   [
        #     {"role": "developer", "content": "Talk like a pirate."},
        #     {
        #         "role": "user",
        #         "content": "How do I check if a Python object is an instance of a class?",
        #     },
        # ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

# --- OpenAI Suggestion Function ---
# async def make_suggestion(client_id: str, text: str):
#     print(f"🤖 Sending to GPT-4o for analysis for {client_id}...")
#     prompt = "You're assisting a user during a meeting. Summarize the conversation and also provide possible answers to questions. Format the answer as a markdown list or paragraph.Conversation: [Question] [Answer] {Answer}"

#     # Append to conversation history for the specific client
#     if client_id not in conversation_history:
#         conversation_history[client_id] = []
#     conversation_history[client_id].append(text)

#     # Use last N turns for context (e.g., last 5 entries)
#     context_text = "\n\n".join(conversation_history[client_id][-5:])

#     messages = [
#         {"role": "system", "content": prompt},
#         {"role": "user", "content": context_text}
#     ]

#     try:
#         completion = await client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages
#         )
#         response_content = completion.choices[0].message.content
#         print(f"GPT-4o response for {client_id}: {response_content}")
#         return response_content
#     except Exception as e:
#         print(f"Error calling OpenAI API for {client_id}: {e}")
#         return "Error generating suggestion."

# --- HTML Content for Client ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>FastAPI Raw Audio WebSocket</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        #messages { list-style-type: none; padding: 0; }
        #messages li { padding: 8px; margin-bottom: 5px; background-color: #f0f0f0; border-radius: 5px; }
        button { padding: 10px 15px; margin: 5px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }
    </style>
</head>
<body>
    <h1>FastAPI Raw Audio WebSocket</h1>
    <button id="startButton">Start Microphone</button>
    <button id="stopButton" disabled>Stop Microphone</button>
    <ul id="messages"></ul>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const messages = document.getElementById('messages');
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');

        let mediaRecorder;
        let audioChunks = [];
        let intervalId; // To send chunks periodically
        const sendInterval = 1000; // Send audio every 1000 ms (1 second)

        ws.onopen = function(event) {
            console.log("WebSocket connection opened:", event);
            addMessage('<em>Connected to server</em>');
            startButton.disabled = false;
        };

        ws.onmessage = function(event) {
            var message = event.data;
            addMessage('Server: ' + message); // Display whatever server sends back
        };

        ws.onclose = function(event) {
            console.log("WebSocket connection closed:", event);
            addMessage('<em>Disconnected from server</em>');
            startButton.disabled = false;
            stopButton.disabled = true;
            stopRecording(); // Ensure recording stops if WS closes
        };

        ws.onerror = function(event) {
            console.error("WebSocket error:", event);
            addMessage('<li style="color: red;"><em>WebSocket error!</em></li>');
            startButton.disabled = false;
            stopButton.disabled = true;
            stopRecording();
        };

        startButton.onclick = async function() {
            startButton.disabled = true;
            stopButton.disabled = false;
            addMessage('<em>Starting microphone...</em>');

            try {
                // Request microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                // Create an AudioContext and AudioWorkletNode for raw float32 samples at 16kHz
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                await audioContext.audioWorklet.addModule('audio-processor.js'); // Load custom AudioWorklet

                const microphone = audioContext.createMediaStreamSource(stream);
                const audioProcessor = new AudioWorkletNode(audioContext, 'audio-data-processor');

                microphone.connect(audioProcessor);
                audioProcessor.connect(audioContext.destination); // Connect to destination to keep it alive

                // Start sending data from the AudioWorkletNode
                audioProcessor.port.onmessage = (event) => {
                    // event.data will be the raw float32 PCM data at 16kHz
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data);
                    }
                };

                addMessage('<em>Microphone started. Sending audio data.</em>');

            } catch (err) {
                console.error('Error accessing microphone:', err);
                addMessage('<li style="color: red;"><em>Error accessing microphone: ' + err.message + '</em></li>');
                startButton.disabled = false;
                stopButton.disabled = true;
            }
        };

        stopButton.onclick = function() {
            stopRecording();
            startButton.disabled = false;
            stopButton.disabled = true;
            addMessage('<em>Microphone stopped.</em>');
        };

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
        }

        function addMessage(msg) {
            messages.innerHTML += '<li>' + msg + '</li>';
            messages.scrollTop = messages.scrollHeight; // Auto-scroll to bottom
        }
    </script>
</body>
</html>
"""

# --- AudioWorklet Processor (JavaScript File) ---
# This JavaScript code needs to be served by FastAPI or saved as 'audio-processor.js'
# alongside your main HTML.
# It downsamples and converts audio to raw float32.
audio_processor_js = """
// audio-processor.js (for AudioWorklet)
class AudioDataProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 16000; // Target sample rate
        this.buffer = []; // To accumulate samples
        this.lastUpdateTime = 0;
        this.sendInterval = 1000; // Send data every 1 second (1000ms)

        // Resampling variables (simple linear interpolation)
        this.resampler = null;
        this.resampler_initialized = false;
    }

    // Initialize the resampler lazily when the first input is received
    // This avoids creating it until we know the input sample rate.
    initResampler(inputSampleRate) {
        if (inputSampleRate === this.sampleRate) {
            this.resampler = null; // No resampling needed
            this.resampler_initialized = true;
            return;
        }

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
        this.resampler_initialized = true;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length === 0) {
            return true; // Nothing to process
        }

        const inputChannelData = input[0]; // Assuming mono or taking the first channel

        if (!this.resampler_initialized) {
            // Initialize resampler with the actual input sample rate (context.sampleRate)
            // It is available via AudioWorkletGlobalScope.currentFrame, or assumed from context
            // In process(), context.sampleRate is not directly accessible. It's the context where worklet was created.
            // A more robust resampler would need to know `audioContext.sampleRate`
            // For simplicity, let's assume `audioContext.sampleRate` is known at init or passed.
            // A common browser default is 44100 or 48000.
            // Let's assume inputSampleRate is `this.sampleRate` for now if we cannot dynamically get it.
            // A better way is to pass `audioContext.sampleRate` from the main thread.
            // For now, let's make a simplifying assumption or mock it for first input.
            // A robust solution involves `wasm-resampler` or a more complex JS resampler.
            // For demonstration, let's just use a fixed input sample rate that's common (48000Hz).
            // YOU SHOULD PASS YOUR AudioContext.sampleRate FROM MAIN THREAD IF IT'S NOT 48000Hz.
            this.initResampler(48000); // Assuming 48000Hz input, adjust as needed
        }

        let processedData = inputChannelData;
        if (this.resampler) {
            processedData = this.resampler(inputChannelData);
        }

        // Accumulate processed data
        this.buffer.push(processedData);

        const currentTime = Date.now();
        if (currentTime - this.lastUpdateTime > this.sendInterval) {
            // Concatenate all accumulated buffers
            const totalLength = this.buffer.reduce((acc, val) => acc + val.length, 0);
            const combinedBuffer = new Float32Array(totalLength);
            let offset = 0;
            for (const array of this.buffer) {
                combinedBuffer.set(array, offset);
                offset += array.length;
            }

            // Post the combined buffer to the main thread
            this.port.postMessage(combinedBuffer.buffer, [combinedBuffer.buffer]); // Transferable
            this.buffer = []; // Clear the buffer
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

    # Initialize buffer for this client
    audio_buffers[client_id] = io.BytesIO()
    conversation_history[client_id] = [] # Initialize conversation history

    try:
        while True:
            # Receive raw audio bytes data
            audio_data_bytes = await websocket.receive_bytes()

            # Append to client's buffer
            audio_buffers[client_id].write(audio_data_bytes)

            # Check if buffer size exceeds threshold
            if audio_buffers[client_id].tell() >= buffer_threshold:
                print(f"[{client_id}] Processing {audio_buffers[client_id].tell()} bytes of audio...")

                # Reset buffer position to start for reading
                audio_buffers[client_id].seek(0)
                
                # Convert bytes to numpy float32 array
                # Note: The client sends float32, so we directly interpret as float32
                audio_np = np.frombuffer(audio_buffers[client_id].read(), dtype=np.float32)

                # --- Perform Transcription (CPU-bound, consider running in threadpool) ---
                # from fastapi.concurrency import run_in_threadpool
                # transcription_result = await run_in_threadpool(
                #     model.transcribe, audio_np, fp16=False
                # )
                # For demonstration, direct call (can block ASGI event loop if too long)
                try:
                    transcription_result = model.transcribe(audio_np, fp16=False)
                    transcription = transcription_result['text'].strip()
                    print(f"[{client_id}] Transcription: {transcription}")
                    
                    if transcription: # Only process if transcription is not empty
                        await websocket.send_text(f"Transcription: {transcription}")

                        # Get and send suggestion
                        suggestion =  makeSuggestion( transcription)
                        await websocket.send_text("Suggestion: " + markdown.markdown(suggestion))

                except Exception as e:
                    print(f"[{client_id}] Error during transcription or suggestion: {e}")
                    await websocket.send_text("Server Error: Failed to process audio.")
                    import traceback
                    traceback.print_exc()


                # Clear the buffer after processing
                audio_buffers[client_id].seek(0)
                audio_buffers[client_id].truncate(0)

    except WebSocketDisconnect:
        print(f"[-] Client {client_id} disconnected.")
        audio_buffers.pop(client_id, None) # Clean up buffer
        conversation_history.pop(client_id, None) # Clean up conversation
    except Exception as e:
        print(f"Error for client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        if client_id in audio_buffers: # Ensure cleanup on other exceptions
            audio_buffers.pop(client_id, None)
            conversation_history.pop(client_id, None)
        try:
            await websocket.close()
        except RuntimeError:
            pass # Connection might already be closed

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
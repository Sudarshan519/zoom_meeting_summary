from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import openai
import tempfile
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
import markdown
import time
import whisper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
model = whisper.load_model("base")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5050"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Conversation storage
conversation = []
TRANSCRIPT_LOG = f"transcript_log{time.time()}.txt"
previousSuggestion = None

def transcribeWhisper(file):
    """Transcribe audio using Whisper"""
    temp_filename = 'audio.webm'
    with open(temp_filename, "wb") as f:
        file.save(f)
        result = model.transcribe(f.name)
        logger.info(f"Transcribed text: {result['text']}")
    return result['text']

def is_question(text):
    """Check if text is a question"""
    if text.strip().endswith('?'):
        return True
    question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'can', 'is', 'are', 'do', 'does', 'did','?']
    words = text.strip().lower().split()
    return words and words[0] in question_words

def make_suggestion(text):
    """Generate meeting response suggestions using GPT-4"""
    logger.info("Generating suggestion with GPT-4...")
    
    prompt = ("You're assisting a user during a meeting. They just said something, "
             "and you're helping them think of a natural, human-like response they could give next. "
             "Only return what the user might naturally say *next*, not a correction or summary.")
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n\n".join(conversation + [text])}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return completion.choices[0].message.content

@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connection"""
    logger.info('Client connected')
    emit('connection_response', {'data': 'Connected successfully'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info('Client disconnected')

@socketio.on('transcribe_audio')
def handle_transcription(data):
    """Handle audio transcription via WebSocket"""
    try:
        # In a real implementation, you'd receive the audio data differently
        # This is a simplified version
        logger.info('Received audio for transcription')
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
            # In a real app, you'd write the received audio data to the file
            # temp.write(data['audio_data'])
            pass
            
        # Transcribe (simplified)
        transcribed_text = transcribeWhisper(data['file'])
        
        # Add to conversation
        conversation.append(transcribed_text)
        
        # Send back transcription
        emit('transcription_result', {'text': transcribed_text})
        
        # If it's a question, generate a suggestion
        if is_question(transcribed_text):
            suggestion = make_suggestion(transcribed_text)
            html_md = markdown.markdown(suggestion)
            emit('suggestion', {'suggestion': html_md})
            
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('analyze_text')
def handle_analysis(data):
    """Handle text analysis via WebSocket"""
    try:
        question = data.get('question')
        if not question:
            raise ValueError("No question provided")
            
        logger.info(f"Analyzing text: {question}")
        
        # Generate suggestion
        suggestion = make_suggestion(question)
        html_md = markdown.markdown(suggestion)
        
        # Send back result
        emit('analysis_result', {'suggestion': html_md})
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import tempfile
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
import markdown
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

import time
import whisper
model = whisper.load_model("small")
def transcribeWhisper(file):
    name=f'audio.webm'
    with open(name, "wb") as f:
        file.save(f)
        result = model.transcribe(f.name)
        print(result['text'])
    return result['text']
# from google.genai import types
# from google import genai
TRANSCRIPT_LOG = f"transcript_log{time.time()}.txt"
previousSuggestion=None
gemini_key=os.environ.get("GEMINI_API_KEY")
def is_question(text):
    # Check if the text ends with a question mark
    if text.strip().endswith('?'):
        return True
    # Optionally, check if the sentence starts with a common question word
    question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'can', 'is', 'are', 'do', 'does', 'did','?']
    words = text.strip().lower().split()
    if words and words[0] in question_words:
        return True
    return False

def makeSuggestion(text):
    print("🤖 Sending to GPT-4 for analysis...")
    # You are an expert in communication analysis. Analyze this short transcript:
    # - Emotional tone
    prompt ="You're assisting a user during a meeting. They just said something, "
    "and you're helping them think of a natural, human-like response they could give next."
    "Only return what the user might naturally say *next*, not a correction or summary."# f"""You are an bot helping user on meetin with context.Separate conversation and also suggest feasible solutions to help host for feasible solutions.Suggest latest answer at top.Respond like as if you are the host of meeting.Respond only in english"""
    conversation_text = "\n\n".join(conversation)  # Join the conversation list into a single string with line breaks
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

    return completion.choices[0].message.content

def transcribeAudio(file):
    mime=(file.content_type)
    next_api_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={gemini_key}"
    files = {
                    "file": (file.name, file,  mime,)  # Send file itself (name, file object, MIME type)
                }        
    genai_response = requests.post(next_api_url, files=files,stream=True)
    # print(genai_response.json())
    # print(genai_response.json())
    
 
    # text= transcribeWhisper(file)
    # return text
    genai_responses=genai_response.json()
    # print(genai_responses['file']['state'])
    if genai_responses['file']['state']=="ACTIVE":
        data=    {
                    "role": "user",
                    "parts": [
                    {"text": "Generate a transcript of the speech.Also provide the suggestions to improve the conversation."},
                    {
                        "file_data": {
                        "mime_type": mime,
                        "file_uri": genai_responses['file']['uri'],
                        }
                    }
                    ]
                        }
        gemini_url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_key}"

        body_data={
                            "contents":data,
                            "systemInstruction": {
                                "role": "user",
                                "parts": [
                                {
                            "text":"Generate a transcript of the speech.If empty or unaudible return <''>.If music or any other tones then ['Music Described'] Also provide the suggestions to improve the conversation."
                                }
                                ]
                            },
                            "generationConfig": {
                                "temperature": 0.15,
                                "topK": 40,
                                "topP": 0.95,
                                "maxOutputTokens": 8192,
                                "responseMimeType": "text/plain"
                            }
                            }
        ai_response = requests.post(gemini_url,json=body_data)
        response_text = ai_response.json()['candidates'][0]['content']['parts'][0]['text']

        return response_text
        suggestion=makeSuggestion(response_text)
        with open(TRANSCRIPT_LOG, "a") as log:
            # log.write(f"\n--- Chunk {chunk_count} ---\n")
            log.write(f"Suggestions [{time.time()}]:\n{suggestion}\n\n")

         # Convert Markdown to HTML
        # is_a_question=is_question(response_text)
        # if is_a_question:
        #     suggestion=makeSuggestion(response_text)
        #     # previousSuggestion=suggestion
        #     conversation.append(response_text)
            
        # else:
        #     suggestion=''

        return ( response_text+"\n"+"Suggestion: "+suggestion)


        
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY" ),
)


# Load environment variables from the .env file
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})
# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"

# Store transcriptions in memory (you can use a database if needed)
conversation = []
def transcribe_chunk(file_path):

    # Use the OpenAI API to transcribe the audio file
    print(f"🧠 Transcribing {file_path}...")
    with open(file_path, "rb") as audio_file:
        key=os.environ.get("OPENAI_API_KEY")
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": audio_file},
            data={"model": "whisper-1"}
        )
    return response.json().get("text", "")

@app.route('/transcribe', methods=["POST", "OPTIONS"])
def transcribe():
    print('transcribe')
    file = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        # file.save(temp.name)
        # temp.flush()

        try:
            audio_file = open(temp.name, "rb")
            # with open('audio.webm', "wb") as f:
            #     file.save(f)
            transcribp=transcribeWhisper(file)
            suggestion=makeSuggestion(transcribp)

            return jsonify({"text":(transcribp)+"----->Suggestion\n\n"+suggestion})
            audioTranscript=transcribeAudio(file)
            # temp.flush()
            conversation.append(audioTranscript)
            print(audioTranscript)
            audioTranscript=markdown.markdown (audioTranscript)
            
            return jsonify({"text":(audioTranscript)})
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text = transcript['text']
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe", 
                file=audio_file
            )
            # Store the transcription
            conversation.append(transcription.text)
            
            return jsonify({"text": transcription.text})
        finally:
            os.remove(temp.name)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    question = data['question']
    # Construct the input to send to GPT
    conversation_text = "\n\n".join(conversation)  # Join the conversation list into a single string with line breaks
    conversation_text += f"\n\n{question}"
    # print(conversation)
    messages = [
        {"role": "system", "content":"You're assisting a user during a meeting. They just said something, "
                "and you're helping them think of a natural, human-like response they could give next."
                "Only return what the user might naturally say *next*, not a correction or summary."},# "You are a coach helping actors improve their responses during interviews or auditions."},
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
    htmlMd=markdown.markdown (completion.choices[0].message.content)
    with open(TRANSCRIPT_LOG, "r") as log:
        text=log.read()
    return jsonify({"suggestion":htmlMd})
    # suggestion = response['choices'][0]['message']['content'].strip()
    # return jsonify({"suggestion": suggestion})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

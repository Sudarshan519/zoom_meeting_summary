from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import tempfile
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Load environment variables from the .env file
load_dotenv()
app = Flask(__name__)
CORS(app)  # Allow frontend to talk to backend

# Set your OpenAI API key here
# openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"

# Store transcriptions in memory (you can use a database if needed)
conversation = []
def transcribe_chunk(file_path):
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

@app.route('/transcribe/', methods=['POST'])
def transcribe():
    file = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        file.save(temp.name)
        temp.flush()

        try:
            audio_file = open(temp.name, "rb")
            # transcript = openai.Audio.transcribe("whisper-1", audio_file)
            # text = transcript['text']
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
    messages = [
        {"role": "system", "content": "You are a coach helping actors improve their responses during interviews or auditions."},
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

    return jsonify({"suggestion":completion.choices[0].message.content})
    # suggestion = response['choices'][0]['message']['content'].strip()
    # return jsonify({"suggestion": suggestion})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

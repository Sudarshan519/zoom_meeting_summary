# Dockerfile
# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for Whisper (ffmpeg)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy your project's requirements file (create one if you don't have it)
# requirements.txt content:
# fastapi
# uvicorn
# python-socketio[asyncio_server]
# python-engineio
# openai
# numpy
# git+https://github.com/openai/whisper.git
# python-dotenv
# markdown
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
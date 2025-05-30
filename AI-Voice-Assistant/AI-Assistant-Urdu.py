# -*- coding: utf-8 -*-
"""Task1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/132lAYb5m5gSluyftL6VR9LlW01HoaAWL
"""

# Install necessary packages
!pip install git+https://github.com/openai/whisper.git
!pip install transformers accelerate bitsandbytes gtts soundfile

import whisper
from gtts import gTTS
from IPython.display import Audio
import tempfile
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from google.colab import files

from transformers import BitsAndBytesConfig

import torch

# Load Whisper model
whisper_model = whisper.load_model("small")

# Load Alif-1.0-8B-Instruct with 4-bit quantization
model_id = "large-traversaal/Alif-1.0-8B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto"
)

# Create chatbot pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# Upload audio
def upload_audio():
    print("Upload your audio file:")
    uploaded = files.upload()
    for fname in uploaded.keys():
        print(f"Uploaded: {fname}")
        return fname

# Transcribe
def transcribe_audio_file(file_path):
    result = whisper_model.transcribe(file_path)
    text = result.get("text", "").strip()
    print(f"You said: {text}")
    return text

# Generate response with Alif model
def generate_response(prompt, max_new_tokens=100):
    response = chatbot(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3)
    full_text = response[0]["generated_text"]
    return full_text[len(prompt):].strip()

# Speak response
def speak(text, lang='ur'):
    print(f"Assistant: {text}")
    tts = gTTS(text=text, lang=lang)
    filename = "response.mp3"
    tts.save(filename)
    display(Audio(filename, autoplay=True))

def main_colab():
    audio_path = upload_audio()
    command = transcribe_audio_file(audio_path)

    if command.lower() in ["exit", "quit", "stop"]:
        speak("خدا حافظ", lang='ur')
        return

    # Make the prompt clearer and only expect the answer
    prompt = f"### Instruction:\n{command}\n\n### Response:"

    response = chatbot(prompt, max_length=1000, min_length=80, do_sample=True, temperature=0.6)
    text = response[0]['generated_text']

    # Extract only the actual response (after ### Response:)
    if "### Response:" in text:
        answer = text.split("### Response:")[1].strip()
    else:
        answer = text.strip()  # fallback in case marker is missing

    speak(answer, lang='ur')

main_colab()

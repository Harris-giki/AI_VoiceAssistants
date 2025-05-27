import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
import tempfile
import os
import wave

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")

# Initialize TTS engine
tts = pyttsx3.init()

# Hugging Face model details
model_id = "large-traversaal/Alif-1.0-8B-Instruct"

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model with quantization and device mapping
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# Create text generation pipeline
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Audio recording function
def record_audio(duration=5, fs=16000):
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return np.squeeze(audio)

# Save audio numpy array as wav file
def save_wav(audio_data, file_path, fs=16000):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

# Transcribe speech to text using Whisper
def transcribe():
    audio_data = record_audio()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    save_wav(audio_data, temp_path)
    result = whisper_model.transcribe(temp_path)
    os.unlink(temp_path)
    text = result.get("text", "").strip()
    print(f"You said: {text}")
    return text

# Generate response using Alif 8B model pipeline
def generate_response(prompt, max_new_tokens=100):
    responses = chatbot(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3)
    # responses is a list of dicts with key "generated_text"
    full_text = responses[0]["generated_text"]
    # Remove prompt from the generated text for cleaner output:
    response = full_text[len(prompt):].strip()
    return response

# Speak out the response
def speak(text):
    print(f"Assistant: {text}")
    tts.say(text)
    tts.runAndWait()

# Main assistant loop
def main():
    while True:
        command = transcribe()
        if command.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break
        response = generate_response(command)
        speak(response)

if __name__ == "__main__":
    main()

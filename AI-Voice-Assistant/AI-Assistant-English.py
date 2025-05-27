import sounddevice as sd
import numpy as np
import whisper
from gtts import gTTS
from playsound import playsound
import tempfile
import os
import wave
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")

# Hugging Face small language model (CPU-friendly)
model_id = "microsoft/phi-1_5"

# Load tokenizer and model (CPU)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create text generation pipeline on CPU
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
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

# Generate response using Hugging Face model
def generate_response(prompt, max_new_tokens=100):
    responses = chatbot(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3)
    full_text = responses[0]["generated_text"]
    response = full_text[len(prompt):].strip()
    return response

# Speak out the response using gTTS
def speak(text, lang='ur'):  # Use 'ur' for Urdu
    print(f"Assistant: {text}")
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        playsound(fp.name)
    os.remove(fp.name)

# Main assistant loop
def main():
    while True:
        command = transcribe()
        if command.lower() in ["exit", "quit", "stop"]:
            speak("خدا حافظ", lang='ur')  # Goodbye in Urdu
            break
        response = generate_response(command)
        speak(response)

if __name__ == "__main__":
    main()

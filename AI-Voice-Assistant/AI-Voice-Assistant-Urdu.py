from transformers import pipeline
import sounddevice as sd
import numpy as np
import whisper
from gtts import gTTS
import tempfile
import os
import wave
import playsound

# Load Whisper model
whisper_model = whisper.load_model("small")

# Load DistilGPT2 text generation pipeline
generator = pipeline('text-generation', model='distilgpt2')

# Record audio
def record_audio(duration=5, fs=16000):
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return np.squeeze(audio)

def save_wav(audio_data, file_path, fs=16000):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

# Transcribe audio to text
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

# Generate response
def generate_response(prompt):
    results = generator(prompt, max_length=100, num_return_sequences=1)
    return results[0]["generated_text"].strip()

# Speak response using gTTS
def speak(text):
    print(f"Assistant: {text}")
    tts = gTTS(text=text, lang="ur")  # Use "en" for English, "ur" for Urdu
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_path)
    playsound.playsound(temp_path)
    os.remove(temp_path)

# Main loop
def main():
    while True:
        command = transcribe()
        if command.lower() in ["exit", "quit", "stop"]:
            speak("خدا حافظ!")
            break
        response = generate_response(command)
        speak(response)

if __name__ == "__main__":
    main()

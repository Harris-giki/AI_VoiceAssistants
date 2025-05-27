import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
from gpt4all import GPT4All
import tempfile
import os
import wave

# Load Whisper model
whisper_model = whisper.load_model("small")  # You can use "tiny", "base", "small", etc.

# Initialize TTS engine
tts = pyttsx3.init()

# Initialize GPT4All
gpt = GPT4All(model_name="qwen2-1_5b-instruct-q4_0.gguf")  # Adjust to your model file path if needed

# Record audio from mic and return raw numpy array
def record_audio(duration=5, fs=16000):
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return np.squeeze(audio)

# Save numpy audio array as a .wav file
def save_wav(audio_data, file_path, fs=16000):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

# Record, save, transcribe, and return text
def transcribe():
    audio_data = record_audio()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()  # Close the file so Whisper can access it

    save_wav(audio_data, temp_path)
    result = whisper_model.transcribe(temp_path)
    os.unlink(temp_path)  # Now it's safe to delete the file

    text = result.get("text", "").strip()
    print(f"You said: {text}")
    return text

# Speak the response using pyttsx3
def speak(text):
    print(f"Assistant: {text}")
    tts.say(text)
    tts.runAndWait()

# Main loop to run the assistant
def main():
    while True:
        command = transcribe()
        if command.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break
        response = gpt.generate(command)
        speak(response)

if __name__ == "__main__":
    main()

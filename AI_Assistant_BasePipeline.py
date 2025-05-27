import sounddevice as sd
import queue
import vosk
import json
import pyttsx3
from gpt4all import GPT4All

# Load VOSK model (adjust path)
model = vosk.Model("./vosk-model-small-en-us-0.15")
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(bytes(indata))

# Initialize recognizer
rec = vosk.KaldiRecognizer(model, 16000)

# Initialize TTS engine
tts = pyttsx3.init()

# Initialize GPT4All
gpt = GPT4All(model_name="qwen2-1_5b-instruct-q4_0.gguf")  # adjust path/model

def listen():
    print("Say something...")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                text = json.loads(result).get("text", "")
                if text:
                    print(f"You said: {text}")
                    return text

def speak(text):
    print(f"Assistant: {text}")
    tts.say(text)
    tts.runAndWait()

def main():
    while True:
        command = listen()
        if command in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break
        # Query GPT4All model
        response = gpt.generate(command)
        speak(response)

if __name__ == "__main__":
    main()

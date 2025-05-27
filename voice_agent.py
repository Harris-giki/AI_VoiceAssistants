import speech_recognition as sr
import pyttsx3
from datetime import datetime

# Speech Recognition
def get_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_sphinx(audio)  # offline recognition
        print("You said:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return ""
    except sr.RequestError:
        print("Speech recognition service failed.")
        return ""

# Text to Speech
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Command Logic
def respond_to_command(command):
    if "hello" in command:
        return "Hi there!"
    elif "your name" in command:
        return "I'm your offline assistant."
    elif "time" in command:
        return f"The time is {datetime.now().strftime('%I:%M %p')}"
    elif "exit" in command or "quit" in command:
        return "Goodbye!"
    else:
        return "I didn't understand. Please try again."

# Main Loop
if __name__ == "__main__":
    while True:
        command = get_audio()
        if command:
            response = respond_to_command(command)
            speak(response)
            if "exit" in command or "quit" in command:
                break

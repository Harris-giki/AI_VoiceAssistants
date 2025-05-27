import speech_recognition as sr #speech to text converter for python
import pyttsx3 #text to speech converter
from datetime import datetime

# Speech Recognition
def get_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source) #audio is not text; it’s a raw audio data object — basically, the recorded sound captured from your microphone.
                                            #This object holds the actual sound waves (voice recording) in memory, not text yet.



    try:
        text = recognizer.recognize_sphinx(audio)  # offline recognition
        print("You said:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return ""
    except sr.RequestError: #Speech service failed (although Sphinx works offline, this is just in case).
        print("Speech recognition service failed.")
        return ""

# Text to Speech
def speak(text):
    engine = pyttsx3.init() # initializes the text-to-speech engine
    engine.say(text)    # adds speech to the queue
    engine.runAndWait()   # speaks out loud

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

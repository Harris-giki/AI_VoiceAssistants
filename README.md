# AI and Non-AI Voice Assistants

This repository contains two different implementations of voice assistants: one powered by AI and another using traditional speech recognition. These implementations demonstrate different approaches to creating voice-based interactive systems.

## Project Structure

```
├── AI-Voice-Assistant/
│   ├── Silero_Gemini_Kokoro/
│   ├── WithOpenAI.py
│   ├── With_HuggingFace.py
│   └── UrduVoiceAssistant_New.ipynb
└── NonAI-Voice-Assistant/
    ├── code.py
    └── requirements.txt
```

## AI Voice Assistant

The AI-powered voice assistant implementation uses multiple advanced technologies:

- **OpenAI Integration**: Utilizes GPT-3.5-turbo for natural language processing
- **AssemblyAI**: Provides real-time speech-to-text transcription
- **ElevenLabs**: Generates high-quality AI voice responses
- **Multilingual Support**: Includes an Urdu language implementation

### Features
- Real-time speech transcription
- Natural language understanding
- High-quality voice synthesis
- Continuous conversation capability
- Multilingual support

### Requirements
- OpenAI API key
- AssemblyAI API key
- ElevenLabs API key

## Non-AI Voice Assistant

A simpler, offline-capable voice assistant implementation that uses:

- **SpeechRecognition**: For speech-to-text conversion
- **pyttsx3**: For text-to-speech synthesis
- **CMU Sphinx**: For offline speech recognition

### Features
- Works completely offline
- Basic command recognition
- Time-telling capability
- Simple conversation flow
- No API dependencies

### Requirements
- Python 3.x
- Dependencies listed in requirements.txt

## Getting Started

### AI Voice Assistant Setup
1. Clone the repository
2. Install required dependencies
3. Add your API keys in the respective files
4. Run the desired implementation:
   - `WithOpenAI.py` for OpenAI-based assistant
   - `With_HuggingFace.py` for HuggingFace-based assistant
   - `UrduVoiceAssistant_New.ipynb` for Urdu language support

### Non-AI Voice Assistant Setup
1. Clone the repository
2. Install dependencies from requirements.txt
3. Run `code.py`

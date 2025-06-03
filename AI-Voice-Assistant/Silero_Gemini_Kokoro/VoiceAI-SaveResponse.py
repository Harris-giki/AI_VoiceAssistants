import sounddevice as sd
import numpy as np
import queue
import threading
import time
import torch
import soundfile as sf
from transformers import pipeline as hf_pipeline
import google.generativeai as genai
from kokoro import KPipeline
from silero_vad import load_silero_vad, get_speech_timestamps
import os

# --- Configuration ---
sample_rate = 16000
chunk_duration = 2  # Reduced for better responsiveness
chunk_samples = int(sample_rate * chunk_duration)
audio_q = queue.Queue()

# Buffer for continuous audio
audio_buffer = np.array([])
buffer_max_duration = 30  # Keep last 30 seconds
buffer_max_samples = int(sample_rate * buffer_max_duration)

# --- Microphone callback ---
def callback(indata, frames, time_info, status):
    if status:
        print("Microphone error:", status)
    audio_q.put(indata.copy())

# --- Load models ---
print("Loading models...")
try:
    vad_model = load_silero_vad()
    print("✓ VAD model loaded")
except Exception as e:
    print(f"✗ VAD model failed: {e}")
    exit(1)

try:
    asr_pipeline = hf_pipeline("automatic-speech-recognition", 
                              model="openai/whisper-small",
                              device=0 if torch.cuda.is_available() else -1)
    print("✓ ASR model loaded")
except Exception as e:
    print(f"✗ ASR model failed: {e}")
    exit(1)

# Gemini setup
GEMINI_API_KEY = "ENTER_API_HERE"  # Replace with your actual key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gen_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✓ Gemini configured")
except Exception as e:
    print(f"✗ Gemini setup failed: {e}")
    exit(1)

# Kokoro TTS setup
try:
    tts_pipeline = KPipeline(lang_code='a')
    print("✓ TTS model loaded")
except Exception as e:
    print(f"✗ TTS model failed: {e}")
    exit(1)

# --- Improved audio processing ---
def process_audio_chunks():
    global audio_buffer
    last_speech_time = 0
    last_silence_time = 0
    speech_detected = False
    silence_threshold = 1.5  # seconds of silence after speech before processing
    
    while True:
        try:
            # Get audio chunk
            audio_data = audio_q.get(timeout=1.0)
            audio_np = audio_data.flatten()
            
            # Add to buffer
            audio_buffer = np.concatenate([audio_buffer, audio_np])
            
            # Keep buffer size manageable
            if len(audio_buffer) > buffer_max_samples:
                audio_buffer = audio_buffer[-buffer_max_samples:]
            
            # Check current chunk for speech
            max_amp = np.max(np.abs(audio_np))
            current_time = time.time()
            
            if max_amp < 0.005:  # Very quiet
                print(f"Max amplitude: {max_amp:.3f} (too low)")
                if speech_detected:
                    last_silence_time = current_time
                continue
                
            # VAD on current chunk
            try:
                speech_segments = get_speech_timestamps(
                    audio_np, 
                    vad_model, 
                    sampling_rate=sample_rate,
                    threshold=0.3,
                    min_speech_duration_ms=200,
                    min_silence_duration_ms=100,
                    return_seconds=True
                )
                
                print(f"Max amplitude: {max_amp:.3f} | Speech segments: {len(speech_segments)}")
                
                if speech_segments:
                    # Speech detected
                    last_speech_time = current_time
                    if not speech_detected:
                        print("🎤 Started speaking...")
                        speech_detected = True
                else:
                    # No speech in current chunk
                    if speech_detected:
                        last_silence_time = current_time
                
                # Process if we had speech and now have enough silence
                if (speech_detected and 
                    last_silence_time > last_speech_time and 
                    (current_time - last_speech_time) > silence_threshold):
                    
                    print("🔄 Processing speech...")
                    # Use the entire buffer for processing
                    process_speech_buffer()
                    
                    # Reset state
                    audio_buffer = np.array([])
                    speech_detected = False
                    last_speech_time = 0
                    last_silence_time = 0
                    
            except Exception as e:
                print(f"VAD error: {e}")
                continue
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {e}")
            continue

def process_speech_buffer():
    """Process the entire audio buffer for speech"""
    if len(audio_buffer) < sample_rate * 0.5:  # Less than 0.5 seconds
        print("Buffer too short, skipping.")
        return
        
    try:
        # Get speech from entire buffer
        speech_segments = get_speech_timestamps(
            audio_buffer, 
            vad_model, 
            sampling_rate=sample_rate,
            threshold=0.3,
            min_speech_duration_ms=200,
            return_seconds=True
        )
        
        if not speech_segments:
            print("No speech found in buffer.")
            return
            
        print(f"Found {len(speech_segments)} speech segments in buffer")
        
        # Extract and combine all speech
        speech_audio = []
        for seg in speech_segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            if end_sample <= len(audio_buffer):
                speech_audio.append(audio_buffer[start_sample:end_sample])
        
        if not speech_audio:
            print("No valid speech extracted from buffer.")
            return
        
        full_speech = np.concatenate(speech_audio)
        
        # Normalize audio
        if np.max(np.abs(full_speech)) > 0:
            full_speech = full_speech / np.max(np.abs(full_speech)) * 0.8
        
        # Save for transcription
        temp_file = "temp_speech.wav"
        sf.write(temp_file, full_speech, sample_rate)
        
        # Transcription
        print("🔍 Transcribing...")
        result = asr_pipeline(temp_file)
        transcription = result.get("text", "").strip()
        
        if len(transcription) < 3:
            print(f"Transcription too short: '{transcription}'")
            return
            
        print(f"💭 You said: '{transcription}'")
        
        # Generate response
        print("🤖 Generating response...")
        response = gen_model.generate_content(f"Respond conversationally in 1-2 sentences: {transcription}").text
        print(f"🗣️ AI: {response}")
        
        # Text to speech
        print("🔊 Generating speech...")
        for i, (_, _, audio) in enumerate(tts_pipeline(response, voice='af_heart')):
            output_file = f"response_{int(time.time())}_{i}.wav"
            sf.write(output_file, audio, sample_rate)
            print(f"💾 Saved: {output_file}")
            
            # Try to play audio
            try:
                if os.name == 'nt':  # Windows
                    os.system(f'start {output_file}')
                elif os.name == 'posix':  # macOS/Linux
                    os.system(f'afplay {output_file}' if os.system('which afplay') == 0 else f'aplay {output_file}')
            except:
                print("(Could not auto-play audio)")
        
        # Clean up
        try:
            os.remove(temp_file)
        except:
            pass
            
        print("✅ Response complete!\n" + "="*50)
        
    except Exception as e:
        print(f"Speech processing error: {e}")



# --- Start everything ---
print("\n🎙️ Voice Assistant Starting...")
print("Speak clearly into your microphone. The system will process speech after detecting silence.")
print("Press Ctrl+C to exit.\n")

# Start audio processing in separate thread
processing_thread = threading.Thread(target=process_audio_chunks, daemon=True)
processing_thread.start()

# Start microphone stream
try:
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=chunk_samples):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n👋 Exiting...")
except Exception as e:
    print(f"Stream error: {e}")
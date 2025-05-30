import assemblyai as aai
from elevenlabs import generate, stream
import os
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

class AI_Assistant:
    def __init__(self):
        # --- API Key Setup ---
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not aai.settings.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable not set.")

        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set.")
        
        print("Loading Hugging Face model: large-traversaai/Alif-1.0-8B-Instruct...")
        model_id = "large-traversaai/Alif-1.0-8B-Instruct"

        # 4-bit quantization configuration
        # This reduces VRAM/RAM usage but requires `bitsandbytes` and a compatible GPU.
        # If you encounter issues with bitsandbytes, you can remove this config
        # and load the model directly (it will use more memory).
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16 if your GPU supports it
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs = {"quantization_config": quantization_config}
            print("Attempting to load model with 4-bit quantization.")
        except ImportError:
            print("bitsandbytes not installed or compatible. Loading model without 4-bit quantization (will use more RAM/VRAM).")
            model_kwargs = {}
        except Exception as e:
            print(f"Error with bitsandbytes config: {e}. Loading model without 4-bit quantization.")
            model_kwargs = {}


        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
            device_map="auto", # Automatically distributes model layers across available GPUs/CPU
            torch_dtype=torch.float16 if "quantization_config" in model_kwargs else torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32 # Use float16 for quantization, bfloat16 if GPU supports it, else float32
        )

        # Create text generation pipeline
        self.chatbot = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model.device, # Ensure pipeline uses the same device as the model
            # You might need to specify trust_remote_code=True for some custom models
        )
        print(f"Hugging Face model loaded successfully on device: {self.model.device}")
        # --- End Hugging Face LLM Setup ---

        self.transcriber = None

        # --- Conversation History / Prompt ---
        self.full_transcript = [
            {"role": "system", "content": "You are a helpful and friendly voice assistant. Respond concisely and keep answers brief."}
        ]

    # --- Real-time Transcription (AssemblyAI) - remains the same ---
    def start_transcription(self):
        """Starts the real-time transcription session."""
        print("\nStarting real-time transcription...")
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=1000
        )
        try:
            self.transcriber.connect()
            microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
            self.transcriber.stream(microphone_stream)
        except Exception as e:
            print(f"Error connecting to AssemblyAI transcriber: {e}")
            self.stop_transcription()

    def stop_transcription(self):
        """Stops the real-time transcription session."""
        if self.transcriber:
            print("Stopping transcription...")
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print(f"AssemblyAI Session ID: {session_opened.session_id}")

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(f"User (interim): {transcript.text}", end="\r")

    def on_error(self, error: aai.RealtimeError):
        print(f"AssemblyAI Error: {error}")

    def on_close(self):
        print("Transcription session closed.")

    # --- LLM Response Generation (Hugging Face Transformers) ---
    def generate_ai_response(self, transcript):
        """Generates AI response using the local Hugging Face LLM and then synthesizes audio."""
        self.stop_transcription()

        user_message = {"role": "user", "content": transcript.text}
        self.full_transcript.append(user_message)
        print(f"\nUser: {transcript.text}")

        try:
            # Format the conversation for the LLM using the tokenizer's chat template
            # This is crucial for instruction-tuned models to understand the turns correctly.
            # `apply_chat_template` returns a list of token IDs, we decode it to a string for the pipeline.
            # Make sure add_generation_prompt=True if you want the template to add the assistant's opening.
            input_ids = self.tokenizer.apply_chat_template(
                self.full_transcript,
                tokenize=False, # We want a string, not token IDs
                add_generation_prompt=True # Add the special tokens to prompt for assistant's turn
            )

            # Generate response using the pipeline
            response = self.chatbot(
                input_ids,
                max_new_tokens=1000, # Maximum number of new tokens to generate
                do_sample=True,      # Use sampling for more creative responses
                temperature=0.7,     # Controls randomness of responses
                top_p=0.9,           # Only sample from top p probability mass
                # Add stop sequences if the model has specific ones (e.g., for end of turn)
                # You can find these in the model's config or by experimentation.
                # For `Alif-1.0-8B-Instruct`, typical chat turn end tokens might be needed.
                # Example: stop_sequences=self.tokenizer.convert_tokens_to_ids(['<|endoftext|>', '<|user|>'])
                # However, the pipeline often handles basic stopping.
            )

            # The pipeline returns a list of dictionaries. Get the generated text.
            # It includes the prompt, so we need to extract only the new generation.
            generated_text_with_prompt = response[0]["generated_text"]
            # Find the start of the actual AI response by removing the input_ids (prompt)
            # This requires careful handling, as templates vary.
            # A simpler way is to split by the last assistant tag if the model generates it,
            # or rely on the model to stop correctly.
            # For `apply_chat_template` with `add_generation_prompt=True`,
            # the generated text will start *after* the prompt.
            ai_response_text = generated_text_with_prompt[len(input_ids):].strip()

            # Clean up potential partial generation or unwanted special tokens if they appear
            if self.tokenizer.eos_token and ai_response_text.endswith(self.tokenizer.eos_token):
                ai_response_text = ai_response_text[:-len(self.tokenizer.eos_token)].strip()
            # If the model emits user tokens after assistant, remove them
            if "<|user|>" in ai_response_text:
                ai_response_text = ai_response_text.split("<|user|>")[0].strip()


            print(f"\nAI Assistant: {ai_response_text}")

            assistant_message = {"role": "assistant", "content": ai_response_text}
            self.full_transcript.append(assistant_message)

            self.generate_audio(ai_response_text)

        except Exception as e:
            print(f"Error generating AI response from local LLM: {e}")
            self.generate_audio("I'm sorry, I encountered an error with the local language model. Please check its setup.")
        finally:
            self.start_transcription()
            print(f"\nReal-time transcription: ", end="\r\n")

    # --- Audio Generation (ElevenLabs) - remains the same ---
    def generate_audio(self, text):
        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice="Rachel",
            model_id="eleven_multilingual_v2",
            stream=True
        )
        stream(audio_stream)

async def main():
    ai_assistant = AI_Assistant()
    initial_greeting = "Hello! How can I assist you today?"
    ai_assistant.generate_audio(initial_greeting)
    ai_assistant.start_transcription()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting AI Assistant.")
        ai_assistant.stop_transcription()

if __name__ == "__main__":
    asyncio.run(main())

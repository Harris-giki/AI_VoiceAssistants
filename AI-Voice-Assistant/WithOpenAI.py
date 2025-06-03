import assemblyai as aai
from elevenlabs import generate, stream
from openai import OpenAI

class AI_Assistant:
    def __init(self):
        aai.settings.api_key="Enter API"
        self.openai_client = OpenAI(api_key="api key") ##need an open ai api key here
        self.elevenlabs_api_key="Enter API"
        
        self.transcriber = None
        
        #Prompt
        self.full_transcript = [
            {"role":"system", "content":"You are voice assistant that tends to answer any user query whatsoever."}
        ]
        
        ##step no 2: Real Time Transcription
        
    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate = 16000,
            on_data = self.on_data,
            on_error = self.on_error,
            on_open = self.on_open,
            on_close = self.on_close,
            end_utterance_silence_threshold = 1000 #If the user pauses for 1 second, it is treated as the end of a sentence/utterance, triggering on_data() with a final transcript (RealtimeFinalTranscript).
        )

        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate =16000)
        self.transcriber.stream(microphone_stream)
    
    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)
        return


    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")


    def on_error(self, error: aai.RealtimeError):
        print("An error occured:", error)
        return


    def on_close(self):
        #print("Closing Session")
        return
    
        #step no 3: Pass real time transcript to openAI
        
    def generate_ai_response(self, transcript):
        self.stop_transcription()
        self.full_transcript.append({"role":"user", "content": transcript.text})
        print(f"\nUser: {transcript.text}", end="\r\n")
        
        response = self.openai_client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = self.full_transcript
        )

        ai_response = response.choices[0].message.content ## retrives the text base responses and stores them

        self.generate_audio(ai_response)

        self.start_transcription()
        print(f"\nReal-time transcription: ", end="\r\n")
        
        
        ##step no.4 : Generating audio through elevenlabs
    def generate_audio(self, text):

        self.full_transcript.append({"role":"assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        audio_stream = generate(
            api_key = self.elevenlabs_api_key,
            text = text,
            voice = "Rachel",
            model_id = "eleven_multilingual_v2",
            stream = True
        )

        stream(audio_stream)

greeting = "I hope I was of good assistance!"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()

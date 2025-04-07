import os
from google.cloud import speech

# Set Google credentials (update with your file path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\User\Desktop\KitaHack25\AI Training\KitaHack25.json"

# Initialize Google Speech-to-Text client
client = speech.SpeechClient()

# Function to transcribe an audio file
def transcribe_audio(audio_file):
    with open(audio_file, "rb") as audio:
        content = audio.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Change if needed
        sample_rate_hertz=16000,  # Adjust based on your audio file
        language_code="en-US",  # Change for other languages
    )

    response = client.recognize(config=config, audio=audio)

    # Print transcription
    for result in response.results:
        print("Transcript:", result.alternatives[0].transcript)

# Example usage
audio_path = r"C:\Users\User\Desktop\KitaHack25\AI Training\6295-244435-0037.wav"  # Replace with your actual file
transcribe_audio(audio_path)

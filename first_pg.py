
import csv
import pyaudio
import os
import wave
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
encoder = VoiceEncoder()

EMBEDDINGS_CSV = "speaker_embeddings.csv"

def extract_features(audio_path):
    """Extracts speaker embeddings using the resemblyzer VoiceEncoder."""
    wav = preprocess_wav(Path(audio_path))
    
    embedding = encoder.embed_utterance(wav)
    return embedding

def record_audio():
    """
    Records audio from the microphone and saves it as a temporary WAV file.

    Returns:
        str: Path to the temporary audio file.
    """

    FRAMES_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    seconds = 6  # Duration of recording

    p = pyaudio.PyAudio()

    # starts recording
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("start recording...")

    frames = []
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    print("recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Create a temporary file
    temp_audio_path = "temp.wav"

    wf = wave.open(temp_audio_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return temp_audio_path


def main():
    while True:
        choice = input("Enter your choice (upload/record/exit): ")

        if choice == "upload":
            audio_path = input("Enter path to audio file: ")
            username = input("Enter the username: ")
            embeddings = extract_features(audio_path)
            save_embeddings(embeddings, username)
            print("Embeddings saved successfully!")

        elif choice == "record":
            audio_path = record_audio()
            username = input("Enter the username: ")
            embeddings = extract_features(audio_path)
            save_embeddings(embeddings, username)
            os.remove(audio_path)  # Remove the temporary file
            print("Embeddings saved successfully!")

        elif choice == "exit":
            break
        else:
            print("Invalid choice. Please try again.")


def save_embeddings(embeddings, username):
    """
    Saves speaker embeddings to a CSV file.

    """

    with open(EMBEDDINGS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([username] + embeddings.tolist())

if __name__ == "__main__":
    main()

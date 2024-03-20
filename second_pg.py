import numpy as np
import csv
import os
import wave
import pyaudio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path

EMBEDDINGS_CSV = "speaker_embeddings.csv"
encoder = VoiceEncoder()
def extract_features(audio_path):
    """Extracts speaker embeddings using the resemblyzer VoiceEncoder."""
    wav = preprocess_wav(Path(audio_path))
    
    embedding = encoder.embed_utterance(wav)
    return embedding

def load_embeddings(embeddings_file):
    embeddings = {}
    with open(embeddings_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            username = row[0]
            embeddings[username] = np.array(row[1:], dtype=float)
    return embeddings

def record_audio():
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

def compare_embeddings(embedding1, embedding2, threshold=0.8):
    embedding1 = preprocessing.normalize(embedding1.reshape(1, -1))[0]
    embedding2 = preprocessing.normalize(embedding2.reshape(1, -1))[0]
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
    return similarity >= threshold 

def main():
    embeddings = load_embeddings(EMBEDDINGS_CSV)

    while True:
        choice = input("Enter your choice (upload/record/exit): ")

        if choice == "upload":
            audio_path = input("Enter path to audio file: ")
            embedding = extract_features(audio_path)

        elif choice == "record":
            audio_path = record_audio()
            embedding = extract_features(audio_path)
            os.remove(audio_path)  # Remove the temporary file

        elif choice == "exit":
            break

        else:
            print("Invalid choice. Please try again.")
            continue 

        # Speaker Recognition Logic
        best_speaker = None 
        best_similarity = 0.0

        for stored_username, stored_embedding in embeddings.items():
            similarity = cosine_similarity(embedding.reshape(1, -1), stored_embedding.reshape(1, -1))[0][0]
            if similarity > best_similarity:
                best_similarity = similarity 
                best_speaker = stored_username

        if best_speaker is not None and best_similarity >= 0.7:  # Apply threshold
            print(f"Speaker recognized: {best_speaker} (similarity: {best_similarity:.3f})")
        else:
            print("Speaker not recognized.")

if __name__ == "__main__":
    main()

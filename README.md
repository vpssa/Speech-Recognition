# Speaker Recognition with Resemblyzer

## Description

This project provides a system for speaker recognition using pre-trained models from the Resemblyzer library. Users can record or upload audio samples, and the system will extract speaker embeddings and compare them against a stored dataset for identification.

## Features

* Two main scripts: 
    * `first_pg.py`: Facilitates audio recording/uploading and stores speaker embeddings.
    * `second_pg.py`: Processes new audio samples and checks for matches within the existing speaker dataset.
* Employs Resemblyzer for high-quality speaker embedding generation.

## Prerequisites

* Python 3.10 
* Required packages (listed in `requirements.txt`)

## Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/vpssa/Speech-Recognition

2. Create a virtual environment (recommended):
   conda create -p venv python==3.10 -y
   conda activate venv
   
3. Install setup.py
   python setup.py install

**Usage**
1. Add speaker samples:

Bash
   python first_pg.py

Follow the prompts to record or upload audio files.
Provide a unique identifier (e.g., name) for each speaker.


2. Check for existing speakers:
   python second_pg.py

Record or upload a new audio sample for identification.
The system will indicate if a match is found and display the speaker's identifier.

   
Important Notes
    1. Ensure correct audio file paths when using first_pg.py and second_pg.py.
    2. Consider the performance and security trade-offs discussed in the project documentation for production-level deployment.
    3. For optimal results in noisy environments, explore noise reduction techniques or consider fine-tuning the Resemblyzer model with domain-specific data.

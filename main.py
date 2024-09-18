import torch
import numpy as np
import sounddevice as sd
from models.autoencoder import DenoisingAutoencoder
from utils.audio_utils import chunk_generator
import librosa

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Load or initialize the model
model = DenoisingAutoencoder().to(device)
model.eval()  # Set model to evaluation mode

# Optionally, load pre-trained weights
model.load_state_dict(torch.load('model.pth'))

# Audio configurations
fs = 16000  # Sampling frequency
chunk_duration = 0.1  # Duration of each chunk in seconds
chunk_size = int(fs * chunk_duration)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    # Convert audio input to tensor
    audio_input = indata[:, 0]
    audio_input = librosa.util.fix_length(audio_input, size=1024)
    audio_tensor = torch.from_numpy(audio_input).float().to(device)

    # Normalize audio
    audio_tensor = (audio_tensor - torch.mean(audio_tensor)) / torch.std(audio_tensor)

    # Forward pass through the model
    with torch.no_grad():
        output = model(audio_tensor)

    # Convert tensor back to numpy
    output_audio = output.cpu().numpy()

    # Play the output audio
    sd.play(output_audio, fs)

# Create input stream
with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, blocksize=chunk_size):
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopped.")
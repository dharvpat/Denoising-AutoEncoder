import torch
import librosa
import numpy as np
import soundfile as sf
from models.autoencoder import UNet1D  # Ensure this is the updated version that returns (recon, mu, logvar)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the trained model
model = UNet1D().to(device)
model.load_state_dict(torch.load('model_Hybrid_large-KL-loss.pth', map_location=device))
model.eval()

# Parameters
sr = 16000
fixed_length = 5*sr  # Adjust if needed

# Load and preprocess the test audio
filename = 'p232_009'
file_path = f'./noisy_testset_wav/{filename}.wav'
audio, _ = librosa.load(file_path, sr=sr)

# Trim or pad the audio to fixed_length
if len(audio) > fixed_length:
    audio = audio[:fixed_length]
else:
    audio = np.pad(audio, (0, fixed_length - len(audio)), mode='constant')

# Normalize the audio
audio_norm = (audio - np.mean(audio)) / (np.std(audio) + 1e-7)

# Convert to tensor with batch and channel dimensions
audio_tensor = torch.tensor(audio_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Run inference through the model
with torch.no_grad():
    reconstructed, mu, logvar = model(audio_tensor)

# Remove batch and channel dimensions and convert to numpy array
reconstructed_audio = reconstructed.squeeze().cpu().numpy()

# Save the reconstructed audio to a file
output_path = f'{filename}_reconstructed.wav'
sf.write(output_path, reconstructed_audio, sr)
print(f"Reconstructed audio saved to {output_path}")
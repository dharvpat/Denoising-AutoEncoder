import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.autoencoder import ConvDenoisingAutoencoder
import librosa
import glob
import os
import numpy as np

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Paths to the dataset directories
noisy_train_dir = './noisy_trainset_28spk_wav/'
clean_train_dir = './clean_trainset_28spk_wav/'

noisy_test_dir = './noisy_testset_wav/'
clean_test_dir = './clean_testset_wav/'

# Load file paths
noisy_train_files = glob.glob(os.path.join(noisy_train_dir, '*.wav'))
clean_train_files = glob.glob(os.path.join(clean_train_dir, '*.wav'))

noisy_test_files = glob.glob(os.path.join(noisy_test_dir, '*.wav'))
clean_test_files = glob.glob(os.path.join(clean_test_dir, '*.wav'))

# Pair files
clean_train_files_dict = {os.path.basename(f): f for f in clean_train_files}
paired_noisy_train_files = []
paired_clean_train_files = []

for noisy_file in noisy_train_files:
    filename = os.path.basename(noisy_file)
    if filename in clean_train_files_dict:
        paired_noisy_train_files.append(noisy_file)
        paired_clean_train_files.append(clean_train_files_dict[filename])

clean_test_files_dict = {os.path.basename(f): f for f in clean_test_files}
paired_noisy_test_files = []
paired_clean_test_files = []

for noisy_file in noisy_test_files:
    filename = os.path.basename(noisy_file)
    if filename in clean_test_files_dict:
        paired_noisy_test_files.append(noisy_file)
        paired_clean_test_files.append(clean_test_files_dict[filename])

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, noisy_files, clean_files, fixed_length=16384):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_file = self.noisy_files[idx]
        clean_file = self.clean_files[idx]

        # Load audio
        noisy, _ = librosa.load(noisy_file, sr=16000)
        clean, _ = librosa.load(clean_file, sr=16000)

        # Fixed-length processing
        if len(noisy) > self.fixed_length:
            max_offset = len(noisy) - self.fixed_length
            offset = np.random.randint(0, max_offset)
            noisy = noisy[offset:(offset + self.fixed_length)]
            clean = clean[offset:(offset + self.fixed_length)]
        else:
            noisy = np.pad(noisy, (0, max(0, self.fixed_length - len(noisy))), 'constant')
            clean = np.pad(clean, (0, max(0, self.fixed_length - len(clean))), 'constant')

        # Normalize
        noisy = (noisy - np.mean(noisy)) / (np.std(noisy) + 1e-7)
        clean = (clean - np.mean(clean)) / (np.std(clean) + 1e-7)

        # Convert to tensors
        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)

        return noisy, clean

# Create datasets and dataloaders
train_dataset = AudioDataset(paired_noisy_train_files, paired_clean_train_files)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = AudioDataset(paired_noisy_test_files, paired_clean_test_files)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss, optimizer
model = ConvDenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for noisy, clean in train_dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Forward pass
        outputs = model(noisy)
        loss = criterion(outputs, clean)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for noisy, clean in test_dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(test_dataloader)
    print(f'Validation Loss: {average_val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'model.pth')

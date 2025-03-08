import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.autoencoder import UNet1D  # Ensure your model now returns (reconstruction, mu, logvar)
import librosa
import glob
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
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

# Pair files for training
clean_train_files_dict = {os.path.basename(f): f for f in clean_train_files}
paired_noisy_train_files = []
paired_clean_train_files = []

for noisy_file in noisy_train_files:
    filename = os.path.basename(noisy_file)
    if filename in clean_train_files_dict:
        paired_noisy_train_files.append(noisy_file)
        paired_clean_train_files.append(clean_train_files_dict[filename])

# Pair files for testing
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
# Make sure UNet1D is updated to output (reconstruction, mu, logvar)
model = UNet1D().to(device)

# Define reconstruction losses
def stft_loss(y_pred, y_true):
    n_fft = 1024  # Or your chosen n_fft value
    window = torch.hann_window(n_fft).to(device)
    hop_length = n_fft // 4
    y_pred_stft = torch.stft(y_pred.squeeze(1), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    y_true_stft = torch.stft(y_true.squeeze(1), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    loss = nn.L1Loss()(torch.abs(y_pred_stft), torch.abs(y_true_stft))
    return loss

def combined_loss(y_pred, y_true):
    loss_time = nn.L1Loss()(y_pred, y_true)
    loss_freq = stft_loss(y_pred, y_true)
    return loss_time + loss_freq

def kl_divergence(mu, logvar):
    # Calculate KL divergence between the posterior and prior (standard normal)
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # Apply numerical stability improvements
    # 1. Use mean instead of sum for batch averaging
    # 2. Apply gradient clipping within the function
    # 3. Handle potential NaN or Inf values
    
    # Replace any NaN values with zeros
    kl_loss = torch.where(torch.isnan(kl_loss), torch.zeros_like(kl_loss), kl_loss)
    
    # Clip extremely large values to prevent overflow
    max_val = 1e6  # A large but finite value
    kl_loss = torch.clamp(kl_loss, max=max_val)
    
    return torch.mean(kl_loss)

# Set optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# Hyperparameter for weighting the KL loss
kl_weight = 1e-3  # Adjust this weight based on your experiments

def get_kl_weight(epoch, warmup_epochs, batch, max_weight=1e-3):
    # Gradually increase KL weight from 0 to max_weight during warmup period
    if epoch > warmup_epochs or batch > 5:
        return max_weight * (epoch+1) / warmup_epochs
    else:
        return 0

# Training loop with Kullback leibler and reconstruction loss hybrid for better outputs
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    i = 0
    for noisy, clean in train_dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        noisy = noisy.unsqueeze(1)  # Add channel dimension
        clean = clean.unsqueeze(1)
        
        # Forward pass - expecting model to return (reconstruction, mu, logvar)
        recon, mu, logvar = model(noisy)
        
        warmup_epochs = 0.5 * num_epochs

        #current_kl_weight = get_kl_weight(epoch, warmup_epochs, max_weight=1e-3, batch = i)
        recon_loss = combined_loss(recon, clean)
        kl_loss = kl_divergence(mu, logvar)
        loss = kl_loss * 10 + recon_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        i += 1
        total_loss += loss
        print(f'Batch {i}/{len(train_dataloader)},Recon Loss: {recon_loss}, KL Loss: {kl_loss}, Loss: {total_loss / i:.4f}, Total Loss: {total_loss}')
        
    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for noisy, clean in test_dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            noisy = noisy.unsqueeze(1)  # Add channel dimension
            clean = clean.unsqueeze(1)
            recon, mu, logvar = model(noisy)
            recon_loss = combined_loss(recon, clean)
            kl_loss = kl_divergence(mu, logvar)
            loss = recon_loss + kl_loss * 10
            total_val_loss += loss

    average_val_loss = total_val_loss / len(test_dataloader)
    print(f'Validation Loss: {average_val_loss:.4f}')
    scheduler.step(average_val_loss)
    
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_Hybrid_large-KL-Loss.pth')
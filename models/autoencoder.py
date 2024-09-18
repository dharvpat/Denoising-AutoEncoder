import torch
import torch.nn as nn

class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(ConvDenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)  # Remove channel dimension
        return x

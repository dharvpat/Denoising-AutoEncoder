import torch
import torch.nn as nn

class UNet1D(nn.Module):
    def __init__(self, num_channels=1, base_filters=64):
        super(UNet1D, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(num_channels, base_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters*2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_filters*2, base_filters*4, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(base_filters*4, base_filters*8, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters*8),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_filters*8, base_filters*4, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_filters*8, base_filters*2, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_filters*4, base_filters, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv1d(base_filters*2, num_channels, kernel_size=15, padding=7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Decoder
        d1 = self.dec1(x4)
        d1 = torch.cat([d1, x3], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, x2], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, x1], dim=1)

        out = self.final_conv(d3)
        out = self.tanh(out)
        return out

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

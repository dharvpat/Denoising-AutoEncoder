import torch
import torch.nn as nn

class UNet1D(nn.Module):
    def __init__(self, num_channels=1, base_filters=64):
        super(UNet1D, self).__init__()

        # Encoder
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
        
        # Layers for latent distribution
        self.fc_mu = nn.Conv1d(base_filters*8, base_filters*8, kernel_size=1)
        self.fc_logvar = nn.Conv1d(base_filters*8, base_filters*8, kernel_size=1)

        # Decoder
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Compute latent distribution parameters
        mu = self.fc_mu(x4)
        logvar = self.fc_logvar(x4)
        # Sample latent vector using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decoder using the sampled latent vector z
        d1 = self.dec1(z)
        d1 = torch.cat([d1, x3], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, x2], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, x1], dim=1)

        out = self.final_conv(d3)
        out = self.tanh(out)
        return out, mu, logvar
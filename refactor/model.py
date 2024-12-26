import torch
import torch.nn as nn
import torch.nn.functional as F

class NeDFModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, skips=[4], L=10):
        super(NeDFModel, self).__init__()
        self.skips = skips
        self.L = L
        self.input_dim = input_dim
        self.encoded_dim = input_dim + 2 * input_dim * L

        # Input layer
        self.layers = nn.ModuleList([nn.Linear(self.encoded_dim, hidden_dim)])

        # Hidden layers with skip connections
        for i in range(7):
            if i + 1 in skips:
                self.layers.append(nn.Linear(hidden_dim + self.input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Final layer for distance prediction
        self.output_layer = nn.Linear(hidden_dim, 1)

    def positional_encoding(self, x):
        """Applies NeRF's exact positional encoding to the input.
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3].
        Returns:
            torch.Tensor: Encoded tensor of shape [B, 6L + 3].
        """
        encoded = [x]
        for i in range(self.L):
            frequency = 2.0 ** i * torch.pi
            encoded.append(torch.sin(frequency * x))
            encoded.append(torch.cos(frequency * x))
        return torch.cat(encoded, dim=-1)

    def forward(self, x):
        encoded_x = self.positional_encoding(x)
        input_x = x  # Use raw input points for skip connections
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                encoded_x = torch.cat([input_x, encoded_x], dim=-1)
            encoded_x = F.relu(layer(encoded_x))
        return F.relu(self.output_layer(encoded_x))
    
def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, 0.0, 1e-2)  # Small uniform initialization
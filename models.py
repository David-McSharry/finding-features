import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearProbe(nn.Module):
    def __init__(self, activation_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(activation_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class NonLinearProbe(nn.Module):
    def __init__(self, activation_dim, output_dim):
        super(NonLinearProbe, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(activation_dim, 512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(512, 512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(512, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class SAE(nn.Module):
    """
    Sparse AutoEncoder
    """

    def __init__(
        self, input_size: int, n_dict_components: int, init_decoder_orthogonal: bool = True
    ):
        """Initialize the SAE.

        Args:
            input_size: Dimensionality of input data
            n_dict_components: Number of dictionary components
            init_decoder_orthogonal: Initialize the decoder weights to be orthonormal
        """

        super().__init__()
        # self.encoder[0].weight has shape: (n_dict_components, input_size)
        # self.decoder.weight has shape:    (input_size, n_dict_components)

        self.encoder = nn.Sequential(nn.Linear(input_size, n_dict_components, bias=True), nn.ReLU())
        self.decoder = nn.Linear(n_dict_components, input_size, bias=True)
        self.n_dict_components = n_dict_components
        self.input_size = input_size

        if init_decoder_orthogonal:
            # Initialize so that there are n_dict_components orthonormal vectors
            self.decoder.weight.data = nn.init.orthogonal_(self.decoder.weight.data.T).T

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass input through the encoder and normalized decoder."""
        c = self.encoder(x)
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)
        return x_hat, c

    @property
    def dict_elements(self):
        """Dictionary elements are simply the normalized decoder weights."""
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device

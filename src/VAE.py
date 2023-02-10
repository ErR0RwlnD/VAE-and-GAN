from abc import ABC, abstractmethod

import torch
from torch import nn
from torchsnooper import snoop


class BaseVAE(ABC, nn.Module):

    @abstractmethod
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def reparameterize(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class LinearVAE(BaseVAE):
    def __init__(self, encoder, decoder, encode_size=128, Z_size=128):
        super().__init__()
        self.encoder = encoder
        self.mu_coder = nn.Linear(encode_size, Z_size)
        self.var_coder = nn.Linear(encode_size, Z_size)
        self.decoder = decoder

    def encode(self, X):
        code = self.encoder(X)
        return self.mu_coder(code), self.var_coder(code)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, Z):
        return self.decoder(Z)

    # @snoop
    def forward(self, X):
        mu, log_var = self.encode(X)
        Z = self.reparameterize(mu, log_var)
        return self.decode(Z), mu, log_var

from abc import ABC, abstractmethod

import torch
from torch import nn


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
        return mu + std * eps

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        mu, log_var = self.encode(X)
        Z = self.reparameterize(mu, log_var)
        return self.decode(Z), mu, log_var

    def sample(self, Z):
        return self.decode(Z)


class MultiLinearVAE(BaseVAE):
    # def __init__(self, encoder: nn.ModuleList, decoder: nn.ModuleList, encode_size:int=128, Z_size:int=128):
    # not use list of Sequential because pytorch has no map like function, catting all input as one chunk runs faster
    def __init__(self, encoder, decoder: nn.ModuleList, encode_size=128, Z_size=128):
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
        return mu + std * eps

    def decode(self, Z):
        output = []
        for dec in self.decoder:
            output.append(dec(Z))
        return output

    def forward(self, X):
        mu, log_var = self.encode(X)
        Z = self.reparameterize(mu, log_var)
        return self.decode(Z), mu, log_var

    def sample(self, Z):
        return self.decode(Z)

    def infer(self, X):
        return self.encoder(X)

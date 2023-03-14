from abc import ABC, abstractmethod

import numpy as np
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

    def reconstruct(self, X):
        mu, _ = self.encode(X)
        return self.decode(mu)

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


class StaticGau(nn.Module):
    def __init__(self, in_channels, out_channels, std=1):
        super().__init__()
        self.encoder = nn.Linear(in_channels, out_channels)
        # self.selector = nn.Linear(in_channels, out_channels)
        self.std = std

    def forward(self, X):
        code = self.encoder(X)
        # selector = torch.softmax(self.selector(X), dim=-1)
        return code + self.std * torch.randn_like(code)
        # return torch.mul((code + self.std * torch.randn_like(code)), selector)


class SGAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size: list[int]):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(StaticGau(input_size, latent_size[0]))
        for i in range(len(latent_size) - 1):
            self.net.append(StaticGau(latent_size[i], latent_size[i + 1]))
        self.net.append(nn.Linear(latent_size[-1], output_size))

    def forward(self, X):
        code = X
        for net in self.net:
            code = net(code)
        return torch.sigmoid(code)

    def sample(self, Z, start_layer):
        code = Z
        for net in self.net[start_layer:]:
            code = net(code)
        return torch.sigmoid(code)


class WideAE(nn.Module):
    def __init__(self, encoder, decoder, selector_input, latent_nums=16, latent_dim=32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mu = nn.Parameter(torch.randn((latent_nums, latent_dim)))
        self.log_var = nn.Parameter(torch.randn((latent_nums, latent_dim)))
        self.selector = nn.Linear(selector_input, latent_nums)
        self.latent_nums = latent_nums

    def forward(self, X):
        precode = self.encoder(X)
        selector = torch.softmax(self.selector(precode), dim=-1)
        batch_size, latent_nums = selector.shape
        latents = self.reparameterize(self.mu, self.log_var).repeat(batch_size, 1, 1).view(batch_size * latent_nums, -1)
        selector = selector.view(-1)
        latents = torch.transpose(latents, 0, 1)
        postcode = torch.mul(selector, latents).transpose(0, 1).view(batch_size, latent_nums, -1)
        postcode = torch.sum(postcode, dim=-2)
        return self.decoder(postcode)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def sample(self, Z: int = None):
        if Z is not None:
            idx = Z
        else:
            idx = np.random.randint(0, self.latent_nums)
        latents = self.reparameterize(self.mu[idx, :], self.log_var[idx, :])
        return self.decoder(latents)


class LinearAE(nn.Module):
    def __init__(self, encoder, decoder, encode_size=128):
        super().__init__()
        self.encoder = encoder
        self.mu_coder = nn.Sequential(nn.Linear(encode_size, 1),
                                      nn.Sigmoid())
        self.decoder = decoder

    def encode(self, X):
        code = self.encoder(X)
        return self.mu_coder(code)

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        mu = self.encode(X)
        return self.decode(mu), mu

    def reconstruct(self, X):
        mu = self.encode(X)
        return self.decode(mu)

    def sample(self, Z):
        return self.decode(Z)

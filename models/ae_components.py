import torch
import torch.nn as nn
import torch.nn.functional as F


class AE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activations, dropout_prob=None):
        super(AE_Encoder, self).__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim, activation in zip(hidden_dims, activations):
          if self._get_activation(activation) == nn.ReLU():
            layers.append(nn.Linear(current_dim, hidden_dim))
            if dropout_prob:
                layers.append(nn.Dropout(dropout_prob))
            layers.append(self._get_activation(activation))
            current_dim = hidden_dim
          else:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout_prob:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.encoder(x)


class AE_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activations, dropout_prob=None):
        super(AE_Decoder, self).__init__()
        layers = []
        current_dim = latent_dim

        # Reverse the order of these lists to keep the autoencoder symmetric
        hidden_dims.reverse()
        activations.reverse()

        for hidden_dim, activation in zip(hidden_dims, activations):
          # In the case of RELU can apply the activatio before dropout
          if self._get_activation(activation) == nn.ReLU():
            layers.append(nn.Linear(current_dim, hidden_dim))
            if dropout_prob:
                layers.append(nn.Dropout(dropout_prob))
            layers.append(self._get_activation(activation))
            current_dim = hidden_dim
          else:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout_prob:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(self._get_activation(activations[-1]))

        self.decoder = nn.Sequential(*layers)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.decoder(x)

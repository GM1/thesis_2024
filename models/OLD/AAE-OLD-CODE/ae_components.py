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


class ConstantDispersionLayer(nn.Module):
    """
    An identity layer that allows injecting extra parameters
    such as dispersion into PyTorch models.
    """
    def __init__(self):
        super(ConstantDispersionLayer, self).__init__()

    def build(self, input_shape):
        """
        Builds the layer by initializing the dispersion parameter `theta`.
        """
        self.theta = nn.Parameter(torch.zeros(1, input_shape[1]))
        self.theta_exp = torch.clamp(torch.exp(self.theta), min=1e-3, max=1e4)

    def forward(self, x):
        """
        Forward pass that simply returns the input `x`.
        """
        return x

    def compute_output_shape(self, input_shape):
        """
        Returns the input shape as the output shape.
        """
        return input_shape

class SliceLayer(nn.Module):
    """
    A layer that slices the input list and returns the element at the specified index.
    """
    def __init__(self, index):
        super(SliceLayer, self).__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass that returns the element at the specified index from the input list `x`.
        """
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        """
        Returns the shape of the element at the specified index in the input shape list.
        """
        return input_shape[self.index]

class ElementwiseDense(nn.Module):
    """
    A dense layer that performs element-wise multiplication of the input with a trainable weight.
    """
    def __init__(self, units, use_bias=True, activation=None):
        super(ElementwiseDense, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.activation = activation

    def build(self, input_shape):
        """
        Builds the layer by initializing the kernel and bias (if used) weights.
        """
        input_dim = input_shape[-1]
        assert (input_dim == self.units) or (self.units == 1), \
               "Input and output dims are not compatible"

        self.kernel = nn.Parameter(torch.ones(self.units))
        if self.use_bias:
            self.bias = nn.Parameter(torch.ones(self.units))
        else:
            self.bias = None

    def forward(self, inputs):
        """
        Forward pass that performs element-wise multiplication and adds bias (if used).
        """
        output = inputs * self.kernel
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

# Lambda layers in PyTorch
def nan2zero(x):
    """
    Replaces NaN values in the tensor `x` with zeros.
    """
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def colwise_multi_layer(l):
    """
    Performs column-wise multiplication of the first tensor in the list `l` with the reshaped second tensor.
    """
    return l[0] * l[1].reshape(-1, 1)
import json
import torch
import torch.nn as nn


class AAE(nn.Module):
    def __init__(self, config_path):
        super(AAE, self).__init__()
        config = self.load_json_config(config_path)
        self.encoder = self.build_network(config["encoder"]["layers"])
        self.decoder = self.build_network(config["decoder"]["layers"])
        self.discriminator = self.build_network(config["discriminator"]["layers"])

    @staticmethod
    def forward(component, x):
        return component(x)

    @staticmethod
    def load_json_config(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
        return config

    @staticmethod
    def build_network(layers_config):
        layers = []
        for layer in layers_config:
            layer_type = layer.pop("type")
            if layer_type == "Linear":
                layers.append(nn.Linear(**layer))
            elif layer_type == "ReLU":
                layers.append(nn.ReLU())
            elif layer_type == "Sigmoid":
                layers.append(nn.Sigmoid())
            elif layer_type == "Tanh":
                layers.append(nn.Tanh())
            # Add more layers as needed
        return nn.Sequential(*layers)



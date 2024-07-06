import json
import torch
import torch.nn as nn

def load_json_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

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


class AAEFromConfig(nn.Module):
    def __init__(self, config_path):
        super(AAEFromConfig, self).__init__()
        config = load_json_config(config_path)
        self.encoder = build_network(config["encoder"]["layers"])
        self.decoder = build_network(config["decoder"]["layers"])
        self.discriminator = build_network(config["discriminator"]["layers"])

    def forward(self, x):
        return self.network(x)


# Initialize networks from JSON config
aae = AAEFromConfig("thesis_2024\\model_configurations\\model0_aae.json")
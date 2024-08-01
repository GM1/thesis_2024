import datetime
import time
import os
import torch
import torch.nn
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from ae_components import *

class AE():
    def __init__(self, scenario, 
                 n_epoch, 
                 dataloader_noise, 
                 dataloader_normal, 
                 data_tensor, 
                 dataset_path, 
                 dropout_prob=None):
        self.scenario = scenario
        self.n_epoch = n_epoch
        self.epochs = range(1, self.n_epoch + 1)
        self.dropout_prob = dropout_prob
        self.dataloader_noise = dataloader_noise
        self.dataloader_normal = dataloader_normal
        self.data_tensor = data_tensor
        self.dataset_path = dataset_path
        self.encoder = None
        self.decoder = None
        self.time_signture = datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
        self._build_ae()
        self.learning_rate = 0.0002
        self.gen_rec_losses = []
        self.gen_bce_losses = []
        self.dis_bce_losses = []
        self.training_duration = 0
        self.losses = {}

    @staticmethod
    def set_requires_grad(model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _build_ae(self):
        df = pd.DataFrame(columns=["objective", "batch_size", "epochs", "hidden_dims", "latent_dim",
                            "encoder_activations", "decoder_activations",
                            "mean_reconstruction_loss",
                            "min_reconstruction_loss",
                            "test_reconstruction_loss",
                            "imputation_nrmse",
                            "imputation_mse",
                            "duration", "timestamp"])
        
        input_dim = self.data_tensor.shape[1]  # Number of genes
        hidden_dims = self.scenario["hidden_dims"]
        latent_dim = self.scenario["latent_dim"]
        encoder_activations = self.scenario["encoder_activations"]
        decoder_activations = self.scenario["decoder_activations"]

        self.encoder = AE_Encoder(input_dim, hidden_dims, latent_dim, encoder_activations, self.dropout_prob)
        self.decoder = AE_Decoder(latent_dim, hidden_dims, input_dim, decoder_activations, self.dropout_prob)

        # Send to GPU/TPU if available
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

        structure = "_".join(str(x) for x in hidden_dims)
        # torch.save(encoder, f"/content/drive/MyDrive/thesis_2024/ae_models/encoders/{time_signature}/ae_encoder_{objective}_{dataset}_{structure}_{time_signature}.pt")
        # torch.save(decoder, f"/content/drive/MyDrive/thesis_2024/ae_models/decoders/{time_signature}/ae_decoder_{objective}_{dataset}_{structure}_{time_signature}.pt")

    #     df.loc[-1] = [objective, batch_size, epochs, hidden_dims, latent_dim,
    #                   encoder_activations, decoder_activations,
    #                   losses["mean_reconstruction_loss"],
    #                   losses["min_reconstruction_loss"],
    #                   losses["test_reconstruction_loss"],
    #                   losses["imputation_nrmse"],
    #                   losses["imputation_mse"],
    #                   total_duration, time_signature]  # adding a row
    #     df.index = df.index + 1  # shifting index
    #     df = df.sort_index()  # sorting by index


        # df.to_csv(f"/content/drive/MyDrive/thesis_2024/ae_results_{dataset}_{time_signature}.csv")
    
    
    def train(self, print_info=True, info_frequency=50):
        training_start_time = time.time()
        
        optimizer = optim.RMSprop(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate)

        mse_loss = nn.MSELoss()

        self.encoder.train()
        self.decoder.train()

        for epoch in self.epochs:
            if epoch % info_frequency == 0:
                # Every 10 epochs, start a new timer
                start = time.time()

            for batch_noise, batch_normal in zip(self.dataloader_noise, self.dataloader_normal):
                batch = batch_noise[0].cuda()
                target = batch_normal[0].cuda()
                batch_rows, batch_cols = batch.shape[0], batch.shape[1]

                # train auto encoder (reconstruction phase)
                self.decoder.zero_grad()
                self.encoder.zero_grad()

                # Generate Latent Data
                latent = self.encoder(batch)

                # Train Generator
                optimizer.zero_grad()
                reconstructed = self.decoder(latent)

                g_loss_mse = mse_loss(reconstructed, target)

                g_loss_mse.backward()
                optimizer.step()

                self.gen_rec_losses.append(g_loss_mse.item())

            if epoch % info_frequency == 0 and print_info:
                # Every 10 epochs, measure the duration
                stop = time.time()
                print(f"[{epoch}/{len(self.epochs)}]: \
                      gen_rec_losses: {torch.mean(torch.FloatTensor(self.gen_rec_losses))},\
                      duration: {stop - start}s")

        training_end_time = time.time()

        self.losses = {"mean_gen_rec_losses": torch.mean(torch.FloatTensor(self.gen_rec_losses)).detach().cpu().numpy().max(),
                "min_gen_rec_losses": torch.min(torch.FloatTensor(self.gen_rec_losses)).detach().cpu().numpy().max(),
                }

        self.training_duration = training_end_time - training_start_time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import scipy
import scanpy as sc
import scvi
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import time

from google.colab import drive

drive.mount('/content/drive')


def build_10x_pbmc_dataset():
    """
      Builds the 10x pbmc dataset from the scVAE paper.
      This will allow for direct comparison.
      Thank you scvi-tools!
    """
    filenames = ["b_cells", "cd4_t_helper", "cd34", "cd56_nk", "regulatory_t", "naive_t", "memory_t", "cytotoxic_t",
                 "naive_cytotoxic"]
    adatas = [scvi.data.dataset_10x(f) for f in filenames]

    # 1. Generating the standard vocabulary using pd.concat
    vocab = pd.DataFrame(pd.concat([adata.var for adata in adatas])["gene_ids"])

    vocab_dedupe = vocab.drop_duplicates(subset=["index", "gene_ids"])

    vocab_dedupe.reset_index(inplace=True)

    vocab = vocab_dedupe

    return vocab


# from torchsummary import summary

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activations, dropout_prob):
        super(Encoder, self).__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim, activation in zip(hidden_dims, activations):
            if self._get_activation(activation) == nn.ReLU():
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.Dropout(dropout_prob))
                layers.append(self._get_activation(activation))
                current_dim = hidden_dim
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout_prob))
                current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.encoder(x)


# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activations, dropout_prob):
        super(Decoder, self).__init__()
        layers = []
        current_dim = latent_dim

        # Reverse the order of these lists to keep the autoencoder symmetric
        hidden_dims.reverse()
        activations.reverse()

        for hidden_dim, activation in zip(hidden_dims, activations):
            # In the case of RELU can apply the activatio before dropout
            if self._get_activation(activation) == nn.ReLU():
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.Dropout(dropout_prob))
                layers.append(self._get_activation(activation))
                current_dim = hidden_dim
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout_prob))
                current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(self._get_activation(activations[-1]))

        self.decoder = nn.Sequential(*layers)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.decoder(x)


# Define the Decoder class
class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activations, dropout_prob):
        super(Discriminator, self).__init__()
        layers = []
        current_dim = latent_dim

        # TODO: May back this out as the Discriminator layers may get their own definition
        # Reverse the order of these lists to keep the autoencoder symmetric
        hidden_dims.reverse()

        for hidden_dim, activation in zip(hidden_dims, activations):
            # In the case of RELU can apply the activatio before dropout
            if self._get_activation(activation) == nn.ReLU():
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.Dropout(dropout_prob))
                layers.append(self._get_activation(activation))
                current_dim = hidden_dim
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout_prob))
                current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(self._get_activation(activations[-1]))

        self.decoder = nn.Sequential(*layers)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.decoder(x)


# Define the Autoencoder class which combines Encoder and Decoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, encoder_activations, decoder_activations, dropout_prob):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, encoder_activations, dropout_prob)
        # Hidden dims are reversed in autoencoder block
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, decoder_activations, dropout_prob)

    def forward(self, x):
        latent_dimension = self.encoder(x)
        reconstructed = self.decoder(latent_dimension)
        return reconstructed, latent_dimension

def train(dataloader,
          epochs,
          distribution,
          criterion,
          criterion_gan,
          encoder,
          decoder,
          discriminator,
          opt_recon,
          opt_gen,
          opt_disc
          ):

    rec_losses = []
    reg_losses = []
    dis_losses = []

    training_start_time = time.time()

    for epoch in epochs:
        start = time.time()
        for batch in dataloader:
            # train auto encoder (reconstruction phase)
            decoder.zero_grad()
            encoder.zero_grad()

            X = batch[0].cuda()
            z_sample = encoder(X)
            reconstructed = decoder(z_sample)

            rec_loss = criterion(X, reconstructed)

            rec_losses.append(rec_loss.item())

            rec_loss.backward()

            opt_recon.step()

            # train discriminator
            discriminator.zero_grad()

            z_fake = encoder(X)
            d_fake = discriminator(z_fake)

            loss_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))
            loss_fake.backward()

            # Now draw "real" sample from prior distribution

            # Standard Gaussian distribution
            # z_real = torch.tensor(torch.randn(X.shape[0], latent_dim)).cuda()
            # Binomial np.random.binomial(n=1, p=0.5, size=(64,1000))
            # np.random.binomial(n=1, p=0.5, size=(X.shape[0], latent_dim)
            z_real = torch.tensor(distribution,
                                  dtype=torch.float32).cuda()

            d_real = torch.tensor(torch.ones(X.shape[0], X.shape[1])).cuda()

            loss_real = criterion_gan(discriminator(z_real), d_real)
            loss_real.backward()

            err_real = loss_real.item()
            err_fake = loss_fake.item()

            # err_disc = err_real + err_fake

            opt_disc.step()

            err_disc = err_real + err_fake

            dis_losses.append(err_disc)

            # Train the encoder to trick the discriminator

            encoder.zero_grad()
            z_fake = encoder(X)
            d_fake = discriminator(z_fake)
            # aim to maximise the discriminator loss
            reg_loss = criterion_gan(d_fake, torch.zeros_like(d_fake))

            reg_losses.append(reg_loss.item())

            reg_loss.backward()
            opt_gen.step()

        if epoch % 10 == 0:
            # Every 10 epochs, measure the duration
            stop = time.time()
            print(f"[{epoch}/{len(epochs)}]: loss_rec: {torch.mean(torch.FloatTensor(rec_losses))}, \
      loss_reg: {torch.mean(torch.FloatTensor(reg_losses))}, \
      loss_dis: {torch.mean(torch.FloatTensor(dis_losses))} \
      duration: {stop - start}s")

    training_end_time = time.time()

    total_duration = training_end_time - training_start_time

    return encoder, decoder, discriminator, total_duration

adata = scvi.data.pbmc_dataset()
adata.obs.str_labels.value_counts()

X = adata.X.todense()
y = np.array(adata.obs.str_labels)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

hvg=5000
adata.var_names_make_unique()
# sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=hvg)

data_matrix = X_train

data_tensor = torch.tensor(data_matrix, dtype=torch.float32)

# Create a dataset and dataloader
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

# Now implement a basic autoencoder architecture...
# Initialize the autoencoder
input_dim = data_tensor.shape[1]  # Number of genes
hidden_dims = [1024, 512]
# Middle layer of the autoencoder is referred to as latent_dim
# Middle layer of the autoencoder is referred to as latent_dim
latent_dim = 100
encoder_activations = ['relu', 'relu', 'relu']
decoder_activations = ['relu', 'relu', 'tanh']
discriminator_activations = ['relu', 'relu', 'sigmoid']
dropout_prob = 0.1

encoder = Encoder(input_dim, hidden_dims, latent_dim, encoder_activations, dropout_prob)
decoder = Decoder(latent_dim, hidden_dims, input_dim, decoder_activations, dropout_prob)
autoencoder = Autoencoder(input_dim, hidden_dims, latent_dim, encoder_activations, decoder_activations, dropout_prob)
discriminator = Discriminator(latent_dim, hidden_dims, input_dim, discriminator_activations, dropout_prob)

criterion = nn.MSELoss()
criterion_gan = nn.BCELoss()
criterion_l1 = nn.L1Loss()

# Send to GPU/TPU if available
if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    autoencoder.cuda()
    discriminator = discriminator.cuda()

# Set learning rates
# gen_lr, reg_lr = 0.0006, 0.0008
gen_lr, reg_lr = 0.001, 0.001

# optimizer_decoder = optim.Adam(decoder.parameters(), lr=gen_lr)
opt_recon = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=gen_lr)
opt_gen = optim.SGD(encoder.parameters(), lr=0.01)
opt_disc = torch.optim.SGD(discriminator.parameters(),lr = 0.01)
# optimizer_encoder = optim.Adam(encoder.parameters(), lr=gen_lr)
# autoencoder_solver = optim.Adam(autoencoder.parameters(), lr=reg_lr)
# Q_generator = optim.Adam(decoder.parameters(), lr=reg_lr) # Redundatant because the encoder is the generator
# optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=reg_lr)

epochs = range(1, 1001)
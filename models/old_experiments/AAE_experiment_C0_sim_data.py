# Databricks notebook source
# MAGIC %md
# MAGIC # Experiment Results
# MAGIC Purpose of experiment is to test results with a Negative Binomial Distribution.

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

# Native
import os
import shutil
from datetime import datetime

# Third party
import scanpy as sc
import scvi
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset, random_split
import sklearn

# Custom imports
from aae import AAE
from data_pipeline_functions import PipelineFunctions as pf

# COMMAND ----------

# MAGIC %md
# MAGIC # Functions

# COMMAND ----------

"""
-------------------------------------------------------------------------------
|   PLOTTING FUNCTIONS                                                        |
-------------------------------------------------------------------------------
"""

def run_dimension_reduction_techniques(adata, obs_label_column):
    # Generate dimension reductions on input data
    sc.tl.pca(adata, n_comps=2)
    sc.pl.pca(adata, color=obs_label_column)
    sc.pp.neighbors(adata, n_neighbors=60, n_pcs=2)
    sc.tl.umap(adata)
    sc.tl.tsne(adata, n_pcs=2)

def original_plots(adata, output_file, obs_label_column, figures_directory, show_fig=False):
    # Requires that run_dimension_reduction_techniques has been run beforehand
    # Save input data plots
    fig = sc.pl.umap(adata, color=obs_label_column, title='Original Data - UMAP - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}original_data_scanpy_umap_{output_file}.png", dpi=300, bbox_inches='tight')
    fig = sc.pl.pca(adata, color=obs_label_column, title='Original Data - PCA - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}original_data_scanpy_pca_{output_file}.png", dpi=300, bbox_inches='tight')
    fig = sc.pl.tsne(adata, color=obs_label_column, title='Original Data - t-SNE - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}original_data_scanpy_tsne_{output_file}.png", dpi=300, bbox_inches='tight')


def generate_latent_space_plots(adata_e, noise_file, obs_label_column, figures_directory, show_fig=False):
    # Requires that run_dimension_reduction_techniques has been run beforehand
    # Save latent space plots
    fig = sc.pl.umap(adata_e, color=obs_label_column, title='Latent Space AAE - UMAP - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}latent_aee_data_scanpy_umap_{noise_file}.png", dpi=300, bbox_inches='tight')
    fig = sc.pl.pca(adata_e, color=obs_label_column, title='Latent Space AAE  - PCA - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}latent_aee_data_scanpy_pca_{noise_file}.png", dpi=300, bbox_inches='tight')
    fig = sc.pl.tsne(adata_e, color=obs_label_column, title='Latent Space AAE  - t-SNE - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}latent_aee_data_scanpy_tsne_{noise_file}.png", dpi=300, bbox_inches='tight')


def generate_denoised_plots(adata_d, noise_file, obs_label_column, figures_directory, show_fig=False):
    # Requires that run_dimension_reduction_techniques has been run beforehand
    # Save decoded/denoised data plots
    fig = sc.pl.umap(adata_d, color=obs_label_column, title='Denoised Data AAE - UMAP - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}denoised_data_scanpy_umap_{noise_file}.png", dpi=300, bbox_inches='tight')
    fig = sc.pl.pca(adata_d, color=obs_label_column, title='Denoised Data AAE - PCA - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}denoised_data_scanpy_pca_{noise_file}.png", dpi=300, bbox_inches='tight')
    fig = sc.pl.tsne(adata_d, color=obs_label_column, title='Denoised Data AAE - t-SNE - Scanpy', return_fig=True, show=show_fig)
    fig.savefig(f"{figures_directory}denoised_data_scanpy_tsne_{noise_file}.png", dpi=300, bbox_inches='tight')

# COMMAND ----------

# MAGIC %md
# MAGIC # IO Directories Configuration

# COMMAND ----------

"""
    THIS CELL IS USED TO CONFIGURE THE INPUT DATA
    AND OUTPUT DIRECTORY STRUCTURES FOR THE EXPERIMENT

    ENSURE CORRECT 
"""
# CONSTANT - TIME SIGNATURE IS ALWAYS REAUIRED
time_signature = datetime.now().isoformat(sep="-").replace(":", "~")[:-7]

# VARIABLE - YOUR INPUT DATA DIRECTORY
input_dataset_directory = "/Volumes/kvai_usr_gmahon1/thesis_2024/simulated_datasets/"

# Root for the output directories, all output folders are created here
root_output_directory = "/Volumes/kvai_usr_gmahon1/thesis_2024/"

# VARIABLE - REMEMBER TO NAME YOUR NOTEBOOKS APPROPRIATELY SO YOU CAN TRACK RESULTS MORE EASILY
# This automatically gets the name of your notebook, and uses it to generate the output directory name,
# So, pick a good name...
notebook_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-1]

# List the input datasets
os.listdir(input_dataset_directory)

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment Configurables

# COMMAND ----------

# Manual work required here if adding new data, 
# these correspond to the noise and ground truth data pairings for training and evaluation
# [(noisy_data, target_data)]
experiment_pairs = [("sim_g2_no_dropout", "sim_g2_no_dropout"), 
                    ("sim_g2_dropout_5", "sim_g2_no_dropout"),
                    ("sim_g4_no_dropout", "sim_g4_no_dropout"),
                    ("sim_g4_dropout_1", "sim_g4_no_dropout"),]


                    # ("sim_g6_no_dropout", "sim_g6_no_dropout"),
                    # ("sim_g6_dropout_1", "sim_g6_no_dropout"),
                    # ("sim_g8_no_dropout", "sim_g8_no_dropout"),
                    # ("sim_g8_dropout_1", "sim_g8_no_dropout"),

# Model configuration
model_config = {"hidden_dims": [1024, 512],
              "latent_dim": 512,
              "encoder_activations": ['leaky_relu', 'leaky_relu'],
              "decoder_activations": ['leaky_relu', 'leaky_relu', 'relu'],
              "distribution": "negative_binomial"
              }

# COMMAND ----------

"""
-------------------------------------------------------------------------------
|   EXPERIMENT CONFIGURABLES                                                  |
-------------------------------------------------------------------------------
"""

# Add a brief description of the purpose of the experiment
experiment_description = \
    """Initial experiments, training the denoising autoencoder with
    normal distribution on simulated data with groups of 2,4,6, and 8 cells.
    There are 8 different simulated datasets, each with 2,000 cellsand 200 genes.
    4 of the datasets are clean, and 4 contain noise.
    The purposed of this experiment is to demonstrate an AAE's
    ability to learn to remove the noise in the datasets. 
    """

# In some experiments, we are not interested in the plots.
generate_plots = True

# This appears to make little to no difference, plots will be displayed during training anyways.
show_fig=False

# Whether or not to save the AAE models
save_model = True

# Whether or not to save the anndata to H5AD files, 
# Saves latent space and denoised data
save_adata = True

# Required for Databricks Unity Catalog Environment, as Scanpy cannot write H5AD files
# directly to external storage on Google Cloud. 
# Workaround is to write to local cluster (ephemeral) storage and then copy to persistent external storage.
is_databricks_uc_env = True

# Instantiate a DataFrame to store run metadata
figures_directory = f"{root_output_directory}figures/{notebook_name}-{time_signature}/"
results_directory = f"{root_output_directory}results/{notebook_name}-{time_signature}/"

os.mkdir(figures_directory)
os.mkdir(results_directory)



# Number of epochs for training
n_epoch = 150

# Batch size for training
batch_size = 32

# Name of column that identifies cell type of each cell
obs_label_column = "Group" # "cell_type1", "str_label"

# The number of epochs that elapse before training losses are printed
info_frequency = 10

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Experiment 

# COMMAND ----------


"""
-------------------------------------------------------------------------------
|   RUN EXPERIMENTS                                                           |
-------------------------------------------------------------------------------
"""


df = pd.DataFrame(columns=["experiment_description", "noise_file", "target_file", 
                           "training_duration", "pca_silhouette", 
                           "umap_silhouette", "tsne_silhouette", 
                           "losses", "n_epoch", 
                           "batch_size", "n_genes", 
                           "n_cells", "model_config", "input_dataset_directory", "results_directory"])

for noise_file, target_file in experiment_pairs:
    noisy_adata = sc.read_h5ad(f"{input_dataset_directory}{noise_file}.h5ad")
    target_adata = sc.read_h5ad(f"{input_dataset_directory}{target_file}.h5ad")
    
    pf.preprocess(noisy_adata, target_sum=1e4, log_data=True, normalise=True)
    pf.preprocess(target_adata, target_sum=1e4, log_data=True, normalise=True)

    assert noisy_adata.shape == target_adata.shape, "Input data shapes do not match"

    n_cells = noisy_adata.shape[0]
    n_genes = noisy_adata.shape[1]

    # perform umap, tsne, and pca
    run_dimension_reduction_techniques(target_adata, obs_label_column)

    if generate_plots:
        original_plots(target_adata, target_file, obs_label_column, figures_directory, show_fig)

    # No point in recalculating everything if the files are the same, so only execute when noise and target files are different.
    # perform umap, tsne, and pca on noise_adata iff input files are not the same
    if target_file != noise_file:
        run_dimension_reduction_techniques(noisy_adata, obs_label_column)

    if target_file != noise_file and generate_plots:
        # Generate dimension reductions on input data
        original_plots(noisy_adata, noise_file, obs_label_column, figures_directory, show_fig)

    # Prep data for model
    X_normal = target_adata.X
    X_noise = noisy_adata.X

    y = np.array(target_adata.obs[obs_label_column])
    # X_train_normal, X_test_normal, y_train_normal, y_test_normal = sklearn.model_selection.train_test_split(X_normal, y, test_size=0.2, random_state=42, stratify=y)

    normal_data_tensor = torch.tensor(X_normal, dtype=torch.float32)
    noise_data_tensor = torch.tensor(X_noise, dtype=torch.float32)

    # Create a dataset and dataloader
    normal_dataset = TensorDataset(normal_data_tensor)
    noise_dataset = TensorDataset(noise_data_tensor)

    normal_dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False)
    noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    dataloader_noise = noise_dataloader
    dataloader_normal = normal_dataloader

    # Instantiate model
    model = AAE(model_config, n_epoch, dataloader_noise, dataloader_normal, normal_data_tensor, dataset_path="")

    # Train model
    model.train(info_frequency=info_frequency)

    # Latent and decoded/denoised data
    latent_data = model.encoder(normal_data_tensor.cuda())
    denoised_data = model.decoder(latent_data)

    detached_counts_e = latent_data.detach().cpu().numpy()

    adata_e = sc.AnnData(detached_counts_e,obs=pd.DataFrame(y, columns=[obs_label_column]))


    # Generate decoded/denoised data plots
    decoded = denoised_data
    detached_counts_d = decoded.detach().cpu().numpy()
    adata_d = sc.AnnData(detached_counts_d, obs=pd.DataFrame(y, columns=[obs_label_column]))

    run_dimension_reduction_techniques(adata_e, obs_label_column)
    run_dimension_reduction_techniques(adata_d, obs_label_column)

    if generate_plots:
        generate_latent_space_plots(adata_e, noise_file, obs_label_column, figures_directory, show_fig)

        generate_denoised_plots(adata_d, noise_file, obs_label_column, figures_directory, show_fig)

    # Save silhouette scores
    labels = adata_d.obs["Group"]

    pca_silhouette = sklearn.metrics.silhouette_score(adata_d.obsm["X_pca"], labels)
    umap_silhouette = sklearn.metrics.silhouette_score(adata_d.obsm["X_umap"], labels)
    tsne_silhouette = sklearn.metrics.silhouette_score(adata_d.obsm["X_tsne"], labels)

    losses = model.losses
    training_duration = model.training_duration

    df.loc[-1] = [experiment_description, f"{noise_file}.h5ad", f"{target_file}.h5ad", 
                  training_duration, pca_silhouette, 
                  umap_silhouette, tsne_silhouette, 
                  losses, n_epoch, 
                  batch_size, n_genes, 
                  n_cells, model_config, input_dataset_directory, results_directory] 
    
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index

    # Now save the model

    if save_model:
        if not os.path.exists(f"{results_directory}models/"):
            os.mkdir(f"{results_directory}models/")
        torch.save(model, f"{results_directory}models/{noise_file}-{target_file}.pt")

    if not is_databricks_uc_env and save_adata:
        if not os.path.exists(f"{results_directory}output_datasets/"):
            os.mkdir(f"{results_directory}output_datasets/")

        adata_e.write_h5ad(f"{results_directory}output_datasets/{noise_file}-{target_file}-encoded.h5ad")
        adata_d.write_h5ad(f"{results_directory}output_datasets/{noise_file}-{target_file}-denoised.h5ad")
    
    # Databricks environment workaround
    elif is_databricks_uc_env and save_adata:
        if not os.path.exists(f"{results_directory}output_datasets/"):
            os.mkdir(f"{results_directory}output_datasets/")

        adata_e.write_h5ad(f"{noise_file}-{target_file}-encoded.h5ad")
        adata_d.write_h5ad(f"{noise_file}-{target_file}-decoded.h5ad")

        shutil.move(f"{noise_file}-{target_file}-encoded.h5ad", f"{results_directory}output_datasets/{noise_file}-{target_file}-encoded.h5ad")
        shutil.move(f"{noise_file}-{target_file}-decoded.h5ad", f"{results_directory}output_datasets/{noise_file}-{target_file}-decoded.h5ad")
    

df.to_csv(f"/Volumes/kvai_usr_gmahon1/thesis_2024/results/{notebook_name}-{time_signature}_simulated_data.csv")

# COMMAND ----------



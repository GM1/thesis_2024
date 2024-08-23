# Databricks notebook source
import scanpy as sc
import scvi
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset, random_split
import sklearn

# COMMAND ----------

from ae import AE
from data_pipeline_functions import PipelineFunctions as pf

# COMMAND ----------

batch_size = 256

# adata = scvi.data.pbmc_dataset()

obs_label_column = "str_labels"
cell_types_to_remove = ["B cells", "CD14+ Monocytes", "Dendritic Cells", "FCGR3A+ Monocytes", "Other", "Megakaryocytes", "NK cells"]

# # adata = scvi.data.purified_pbmc_dataset()

cell_types_to_remove = ["Other", "Megakaryocytes"]

reduced_data = pf.remove_cell_types(adata, obs_label_column, cell_types_to_remove)

pf.preprocess(adata=reduced_data, n_top_genes=2000, min_genes=200, min_cells=3, target_sum=1e4, log_data=True, normalise=True)

reduced_subset = reduced_data[:, reduced_data.var["highly_variable"]]

# COMMAND ----------

reduced_subset.random_masked, mask_rows = pf.add_noise(reduced_subset, 0.35)
counts_masked=reduced_subset.random_masked
cellinfo_mask=pd.DataFrame(reduced_subset.obs["str_labels"]) # "str_labels" "cell_types"
geneinfo_mask=pd.DataFrame(reduced_subset.var['gene_symbols']) # adata.var['gene_symbols'] adata.var.reset_index(inplace=False).index

subset_masked = sc.AnnData(counts_masked, obs=cellinfo_mask, var=geneinfo_mask)
subset_masked

sc.tl.pca(subset_masked, n_comps=2)
sc.pl.pca(subset_masked, color='str_labels',title='Masked PBMC data')

# COMMAND ----------

X_normal = reduced_subset.X.todense()
X_masked = reduced_subset.random_masked.todense()

y = np.array(reduced_subset.obs[obs_label_column])
X_train_normal, X_test_normal, y_train_normal, y_test_normal = sklearn.model_selection.train_test_split(X_normal, y, test_size=0.2, random_state=42, stratify=y)
X_train_masked, X_test_masked, y_train_masked, y_test_masked = sklearn.model_selection.train_test_split(X_masked, y, test_size=0.2, random_state=42, stratify=y)

normal_data_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
masked_data_tensor = torch.tensor(X_train_masked, dtype=torch.float32)

# Create a dataset and dataloader
normal_dataset = TensorDataset(normal_data_tensor)
masked_dataset = TensorDataset(masked_data_tensor)

normal_dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False)
masked_dataloader = DataLoader(masked_dataset, batch_size=batch_size, shuffle=False)

# COMMAND ----------

scenario={"hidden_dims": [1024, 512],
              "latent_dim": 512,
              "encoder_activations": ['leaky_relu', 'leaky_relu'],
              "decoder_activations": ['leaky_relu', 'relu'],
              }
n_epoch = 500
dataloader_noise = masked_dataloader
dataloader_normal = normal_dataloader


# COMMAND ----------

model = AE(scenario, n_epoch, dataloader_noise, dataloader_normal, normal_data_tensor, dataset_path="")

# COMMAND ----------

model.train()

# COMMAND ----------

latent_data = model.encoder(masked_data_tensor.cuda())
denoised_data = model.decoder(latent_data)

# COMMAND ----------

detached_counts_e = latent_data.detach().cpu().numpy()
adata_e = sc.AnnData(detached_counts_e,obs=pd.DataFrame(y_train_normal, columns=["str_labels"]))
sc.tl.tsne(adata_e, n_pcs=2)
sc.tl.pca(adata_e, n_comps=2)

# COMMAND ----------

decoded = denoised_data
detached_counts_d = decoded.detach().cpu().numpy()
adata_d = sc.AnnData(detached_counts_d,obs=pd.DataFrame(y_train_normal, columns=["str_labels"]))
sc.tl.tsne(adata_d, n_pcs=2)
sc.tl.pca(adata_d, n_comps=2)

# COMMAND ----------

sc.pl.pca(adata_e, color='str_labels',title='PCA - MSE Adversarial Autoencoder Latent Space')

# COMMAND ----------

sc.pl.tsne(adata_e, color='str_labels',title='t-SNE - MSE Adversarial Autoencoder Latent Space')

# COMMAND ----------

sc.pl.pca(adata_d, color='str_labels',title='PCA - MSE Adversarial Autoencoder Decoded Space')

# COMMAND ----------

sc.pl.tsne(adata_d, color='str_labels',title='t-SNE - MSE Adversarial Autoencoder Decoded Space')

# COMMAND ----------

pip install magic-impute

# COMMAND ----------

import magic
import anndata
import numpy as np
import pandas as pd



# Step 4: Initialize MAGIC
magic_operator = magic.MAGIC()

# Step 5: Apply MAGIC to denoise the data
adata_magic = magic_operator.fit_transform(reduced_subset)

# Step 6: Save the denoised data to a new file
# file_path = 'denoised_pbmc_data.h5ad'
# adata_magic.write(file_path)

# print(f"MAGIC denoising completed and data saved to {file_path}")

# COMMAND ----------

adata_magic

# COMMAND ----------

sc.tl.tsne(adata_magic, n_pcs=2)
sc.tl.pca(adata_magic, n_comps=2)

# COMMAND ----------

sc.pp.neighbors(adata_magic)
sc.tl.umap(adata_magic, n_components=2)

# COMMAND ----------

sc.pl.pca(adata_magic, color='labels',title='PCA - MAGIC')

# COMMAND ----------

sc.pl.umap(adata_magic, color='labels',title='UMAP - MAGIC')

# COMMAND ----------



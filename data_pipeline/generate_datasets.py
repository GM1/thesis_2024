# Databricks notebook source
pip install numpy==1.23.5

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import scanpy as sc
import scvi

import os

# COMMAND ----------

from data_pipeline_functions import PipelineFunctions as pf

# COMMAND ----------

# MAGIC %md # Dataset 1. SCVI - PBMC Dataset Reduced, 2 Cell Types Only

# COMMAND ----------

dataset_name = "scvi_pbmc.h5ad"
obs_label_column = "str_labels"
cell_types_to_remove = ["B cells", "CD14+ Monocytes", "Dendritic Cells", "FCGR3A+ Monocytes", "Other", "Megakaryocytes", "NK cells"]

if dataset_name not in os.listdir("/Volumes/kvai_usr_gmahon1/thesis_2024/raw_datasets/"):
    adata = scvi.data.pbmc_dataset()
    # adata.write_h5ad(filename="/Volumes/kvai_usr_gmahon1/thesis_2024/raw_datasets/" + dataset_name)
else:
    adata = sc.read("/Volumes/kvai_usr_gmahon1/thesis_2024/raw_datasets/" + dataset_name)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# Databricks notebook source
import cellxgene_census
import pandas as pd
import pyspark.pandas as ps
import numpy as np
from typing import List
import os
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse
from typing import Optional, Dict

from .anndata_utils import *
from .download_cxgene_data import *


# My own implementation for cellxgene/build_soma_index.py

import yaml
import json


from .constants import CATALOG, SCHEMA
organism = "mus_musculus"


def preprocess(
    adata: sc.AnnData,
    main_table_key: str = "counts",
    include_obs: Optional[Dict[str, List[str]]] = None,
    N=10000,
    filter_genes_=False, 
    log1p_=False,
    normalize_=False
) -> sc.AnnData:
    """
    Preprocess the data for scBank. This function will modify the AnnData object in place.

    Args:
        adata: AnnData object to preprocess
        main_table_key: key in adata.layers to store the main table
        include_obs: dict of column names and values to include in the main table

    Returns:
        The preprocessed AnnData object
    """
    if include_obs is not None:
        # include only cells that have the specified values in the specified columns
        for col, values in include_obs.items():
            adata = adata[adata.obs[col].isin(values)]

    # filter genes
    if filter_genes_:
        sc.pp.filter_genes(adata, min_counts=(3 / 10000) * N)

    # TODO: add binning in sparse matrix and save in separate datatable
    # preprocessor = Preprocessor(
    #     use_key="X",  # the key in adata.layers to use as raw data
    #     filter_gene_by_counts=False,  # step 1
    #     filter_cell_by_counts=False,  # step 2
    #     normalize_total=False,  # 3. whether to normalize the raw data and to what sum
    #     log1p=False,  # 4. whether to log1p the normalized data
    #     binning=51,  # 6. whether to bin the raw data and to what number of bins
    #     result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    # )
    # preprocessor(adata)

    adata.layers[main_table_key] = adata.X.copy()  # preserve counts
    if normalize_:
        sc.pp.normalize_total(adata, target_sum=1e4)
    
    if log1p_:
        sc.pp.log1p(adata)
    # adata.raw = adata  # freeze the state in `.raw`

    # apply a hard clip to the data for now
    print(
        f"original mean and max of counts: {adata.layers[main_table_key].mean():.2f}, "
        f"{adata.layers[main_table_key].max():.2f}"
    )
    # if isinstance(adata.layers[main_table_key], np.ndarray):
    #     adata.layers[main_table_key] = adata.layers[main_table_key].clip(0, 30)
    # else:  # assume it is a sparse matrix
    #     adata.layers[main_table_key].data = adata.layers[main_table_key].data.clip(0, 30)

    return adata


# COMMAND ----------

# Best to bin while writing the data as the method from scGPT consumes a huge amount of data. Looks like binning() function from preprocess.py is used instead of the binning code from Preprocessor class.

def _digitize(x: np.ndarray, bins: np.ndarray, side="one") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_digits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits



def load_base_vocab(organism="mus_musculus"): 
    if spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary"):
        return spark.table(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary")
    else:
        return None

def create_base_vocab(anndata, organism="mus_musculus"):
    vocab = anndata.var[["feature_name", "soma_joinid"]]
    # Used to start at 1, but should actually start at zero
    # vocab.soma_joinid = vocab.soma_joinid.add(1)
    vocab.rename(columns={"feature_name": "gene_name", "soma_joinid": "gene_id"}, inplace=True)
    spark.createDataFrame(vocab).write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary")

# TODO: test this functionality and add celltypes column to cellbank format...
def update_base_vocab(anndata, organism="mus_musculus"):   
    incoming = anndata.var[["feature_name"]]
    incoming.rename(columns={"feature_name": "gene_name"}, inplace=True)
    existing = spark.table(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary").toPandas()
    net_new = set(incoming.gene_name).difference(existing.gene_name)
    print(net_new)
    if net_new:
        where_diff = incoming.gene_name.isin(net_new)

        previous_highest_gene_id = existing.gene_id.max()

        new_genes = incoming[where_diff]
        new_genes.insert(0, 'gene_id', range(previous_highest_gene_id + 1, previous_highest_gene_id + 1 + len(incoming[where_diff])))

        new_vocab = pd.concat([existing, new_genes])
        new_vocab.sort_values(by=["gene_id"], inplace=True)

        spark.createDataFrame(d).write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary")

    else:
        print("INFO: No new genes to add to the vocabulary table")


def check_vocab_exists(organism="mus_musculus"):    
    return spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary")

# COMMAND ----------

def generate_tokens(anndata, ind_map):
    n_rows, n_cols = anndata.X.shape
    new_indices = np.array(
                [ind_map.get(i, -100) for i in range(n_cols)], int
            ) 

    n_rows, n_cols = anndata.X.shape
    indptr = anndata.X.indptr
    indices = anndata.X.indices
    non_zero_data = anndata.X.data

    tokenized_data = {"cell_id": [], "genes": [], "expressions": [], "binned_expressions": []}
    tokenized_data["cell_id"] = list(range(n_rows))

    for i in range(n_rows):  # ~2s/100k cells
        row_indices = indices[indptr[i] : indptr[i + 1]]
    for row in anndata.X:
        row_new_indices = new_indices[row_indices]
        row_non_zero_data = non_zero_data[indptr[i] : indptr[i + 1]]

        match_mask = row_new_indices != -100
        row_new_indices = row_new_indices[match_mask]
        row_non_zero_data = row_non_zero_data[match_mask]

        tokenized_data["genes"].append(row.indices)
        tokenized_data["expressions"].append(row.data)
        tokenized_data["binned_expressions"].append(binning(row))

    return tokenized_data

# COMMAND ----------

def add_metadata(df, cell_type):
    return df.withColumn("celltypes", lit(cell_type))

def anndata_to_cell_bank_test(anndata, cell_type, delta_table="", output_directory=""):
    """
        Function converts an incoming anndata file to a cell bank format. 
        Writes to delta table or parquet files. 
        Returns DataFrame
    """
    if not output_directory:
        raise Exception("Please provide an output directory")
    
    if not delta_table:
        print("WARNING: No delta table name provided, delta table will not be created")

    base_vocab = load_base_vocab()
    incoming_vocab = load_new_vocab(anndata)

    base_vocab.createOrReplaceTempView("base_gene_vocabulary")
    incoming_vocab.createOrReplaceTempView("incoming_gene_vocabulary")
    
    print("Mapping new vocab to old")
    mapped = spark.sql("""SELECT index, gene_id AS base_gene_id 
                            FROM base_gene_vocabulary 
                            INNER JOIN incoming_gene_vocabulary 
                            ON gene_name = feature_name""")
    
    ind_map = {x[0]: x[1] for x in mapped.select("index", "base_gene_id").collect()}

    tokens = generate_tokens(anndata, ind_map)

    tokens = ps.DataFrame.from_dict(tokens).to_spark()

    tokens = add_metadata(tokens, cell_type)

    if delta_table:
        print("Creating delta table")
        tokens.write.format("delta").mode("overwrite").saveAsTable(delta_table) # "kvai_usr_gmahon1.thesis.microglia_sequences_test_data"

    print("Writing to parquet files")
    tokens.write.format("parquet").mode("overwrite").save(output_directory) # "/Volumes/kvai_usr_gmahon1/thesis/test_data/microglia"

    return tokens

# COMMAND ----------

def binning(row, n_bins=51):
    # assert issparse(adata.X)
    # for row in adata.X:
    expressions = row.data
    bins = np.quantile(expressions, np.linspace(0, 1, n_bins - 1))
    return _digitize(expressions, bins) # , bins



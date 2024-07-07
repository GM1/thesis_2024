from pyspark.sql.functions import *
from pyspark.sql.types import *

import pandas as pd
import pyspark.pandas as ps
import scanpy

from pyspark.sql import SparkSession
from .constants import CATALOG, SCHEMA
# from pyspark.context import SparkContext

spark = SparkSession.getActiveSession()



def map_indices_and_return_tokens(anndata, ind_map):
    n_rows, n_cols = anndata.X.shape
    new_indices = np.array(
                [ind_map.get(i, -100) for i in range(n_cols)], int
            ) 
    indptr = anndata.X.indptr
    indices = anndata.X.indices
    non_zero_data = anndata.X.data

    tokenized_data = {"cell_id": [], "genes": [], "expressions": []}
    tokenized_data["cell_id"] = list(range(n_rows))
    for i in range(n_rows):  # ~2s/100k cells
        row_indices = indices[indptr[i] : indptr[i + 1]]
        row_new_indices = new_indices[row_indices]
        row_non_zero_data = non_zero_data[indptr[i] : indptr[i + 1]]

        match_mask = row_new_indices != -100
        row_new_indices = row_new_indices[match_mask]
        row_non_zero_data = row_non_zero_data[match_mask]

        tokenized_data["genes"].append(row_new_indices)
        tokenized_data["expressions"].append(row_non_zero_data)

    return tokenized_data

def load_new_vocab(anndata):
    # Using pandas, PySpark and Pandas API on Spark where appropriate
    incoming_vocab = pd.DataFrame(anndata.var["feature_name"])
    incoming_vocab = incoming_vocab.astype({"feature_name": "object"})

    incoming_vocab.reset_index(inplace=True)
    incoming_vocab = ps.DataFrame(incoming_vocab["feature_name"])
    incoming_vocab.reset_index(inplace=True)
    incoming_vocab.index += 1

    # Load the gene_ids of the new cell data and begin the process of merging the two vocabularies
    return ps.DataFrame(incoming_vocab).to_spark() # Using this (lazy) approach to go around the type casting issue

def load_base_vocab(organism="mus_musculus"): 
    if spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary"):
        return spark.table(f"{CATALOG}.{SCHEMA}.{organism}_gene_vocabulary")
    else:
        return None

def create_base_vocab(anndata, organism="mus_musculus"):
    pass

def anndata_to_cell_bank(anndata, cell_type, delta_table="", output_directory=""):
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
    if not base_vocab:
        print(f"INFO: No base vocabulary found, creating new vocabulary {organism}_gene_vocabulary")
    incoming_vocab = load_new_vocab(anndata)

    base_vocab.createOrReplaceTempView("base_gene_vocabulary")
    incoming_vocab.createOrReplaceTempView("incoming_gene_vocabulary")
    
    print("Mapping new vocab to old")
    mapped = spark.sql("""SELECT index, gene_id AS base_gene_id 
                            FROM base_gene_vocabulary 
                            INNER JOIN incoming_gene_vocabulary 
                            ON gene_name = feature_name""")
    
    ind_map = {x[0]: x[1] for x in mapped.select("index", "base_gene_id").collect()}

    tokens = map_indices_and_return_tokens(anndata, ind_map)

    tokens = ps.DataFrame.from_dict(tokens).to_spark()

    if delta_table:
        print("Creating delta table")
        tokens.write.format("delta").mode("overwrite").saveAsTable(delta_table) # "kvai_usr_gmahon1.thesis.microglia_sequences_test_data"

    print("Writing to parquet files")
    tokens.write.format("parquet").mode("overwrite").save(output_directory) # "/Volumes/kvai_usr_gmahon1/thesis/test_data/microglia"

    return tokens
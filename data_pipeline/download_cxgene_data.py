import cellxgene_census
from typing import List
import os


# My own implementation for cellxgene/build_soma_index.py

import yaml

with open('/Workspace/Repos/gmahon1@kenvue.com/kvai_usr_gmahon1/scDL/data_processing/data_config.yaml') as f:
    data_config = yaml.safe_load(f)

VERSION = data_config["VERSION"]
VALUE_FILTER = data_config["VALUE_FILTER"]

def retrieve_soma_idx(query_name, organism="mus_musculus"):
    """
    This function is used to retrieve cell soma ids from cellxgene census based on the query name
    """

    with cellxgene_census.open_soma(census_version=VERSION) as census:
        cell_metadata = census["census_data"][organism].obs.read(
        value_filter = VALUE_FILTER[query_name],
        column_names = ["soma_joinid"]
    )
    cell_metadata = cell_metadata.concat()
    cell_metadata = cell_metadata.to_pandas()
    return cell_metadata["soma_joinid"].to_list()

def define_partition(partition_idx, id_list, partition_size) -> List[str]:
    """
    This function is used to define the partition for each job

    partition_idx is the partition index, which is an integer, and 0 <= partition_idx <= len(id_list) // MAX_PARTITION_SIZE
    """
    i = partition_idx * partition_size
    return id_list[i:i + partition_size]

def download_partition(partition_idx, query_name, id_list, partition_size, organism="Mus musculus"):
    """
    This function is used to download the partition_idx partition of the query_name
    """
    # define id partition
    #id_list = load2list(query_name, index_dir)
    id_partition = define_partition(partition_idx, id_list, partition_size)
    print(id_partition)
    with cellxgene_census.open_soma(census_version=VERSION) as census:
        adata = cellxgene_census.get_anndata(census,
                                            organism=organism,
                                            obs_coords=id_partition,
                                            )
    # prepare the query dir if not exist
    # query_dir = os.path.join(output_dir, query_name)
    # if not os.path.exists(query_dir):
    #     os.makedirs(query_dir)
    # query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    # adata.write_h5ad(query_adata_path)
    return adata
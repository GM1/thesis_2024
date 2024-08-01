import scvi
import scanpy as sc
import numpy as np
import time
from scipy.sparse import coo_matrix, save_npz, vstack, csr_matrix


class PipelineFunctions:
    @staticmethod
    def preprocess(adata, 
                n_top_genes=2000, 
                min_genes=200, 
                min_cells=3, 
                target_sum=1e4, 
                log_data=True, 
                normalise=True):
        """
            Typical preprocessing pipeline for single cell data
        """
        # Preprocessing
        if min_genes:
            sc.pp.filter_cells(adata, min_genes=200)
        
        if min_cells:
            sc.pp.filter_genes(adata, min_cells=3)
        
        if normalise:
            sc.pp.normalize_total(adata, target_sum=1e4)

        if log_data:
            adata.raw = adata
            sc.pp.log1p(adata)

        if n_top_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)


    # This is a slower method that guarantees a masking rate of gene expressions of 50% per cell
    # uniform distribution
    @staticmethod
    def add_noise_per_cell(subset, mask_fraction):
        """
            Generates noise in dataset by randomly(normal) deleting
            a specified fraction of the expressions. 
            The distribution is applied at the cell level.
        """
        new_indptr = np.rint((subset.X.indptr * (1-mask_fraction)))
        new_indices = np.array([])
        new_data = np.array([])

        mask_indptr = np.array([0])
        mask_rows = np.array([])

        start = time.time()
        for idx,cell in enumerate(subset.X):
            if idx % 1000 == 0:
                print(idx)
            mask_row = np.random.choice(range(len(cell.data)), 
                                        size=round(len(cell.data) 
                                        * mask_fraction), replace=False)
            masked_expressions = cell.data
            masked_expressions[mask_row] = 0
            indices = np.delete(cell.indices, mask_row)
            data = np.delete(masked_expressions, mask_row)

            if len(indices) != (new_indptr[idx+1] - new_indptr[idx]):
                new_indptr[idx+1] = len(indices) + new_indptr[idx]
            new_indices = np.append(new_indices, indices)
            new_data = np.append(new_data, data)
            # Add code to track the indices of the masked rows per cell.
            if idx != 0:
                mask_indptr = np.append(mask_indptr, len(mask_row) + mask_indptr[idx - 1])
            else: 
                mask_indptr = np.append(mask_indptr, len(mask_row))

            mask_rows = np.append(mask_rows, mask_row)

        end = time.time()
        print(end - start)

        return csr_matrix((new_data, new_indices, new_indptr)), mask_rows, mask_indptr

    
    @staticmethod
    def add_noise(subset, mask_fraction):
        """
            Generates noise in dataset by randomly(normal) deleting
            a specified fraction of the expressions. 
            The distribution is applied across the whole dataset
        """
        new_indptr = np.rint((subset.X.indptr * (1-mask_fraction)))
        new_indices = np.array([])
        new_data = np.array([])

        start = time.time()
        mask_length = len(subset.X.indices) - int(new_indptr[-1])
        mask_row = np.random.choice(range(len(subset.X.data)), 
                                    size=round(len(subset.X.data) 
                                    * mask_fraction), replace=False)
        mask_row = mask_row[:mask_length]
        masked_expressions = subset.X.data
        new_indices = np.delete(subset.X.indices, mask_row)
        new_data = np.delete(masked_expressions, mask_row)

        return csr_matrix((new_data, new_indices, new_indptr)), mask_row


    @staticmethod
    def remove_cell_types(adata, 
                        obs_label_column,
                        cell_types_to_remove):
        
        modified = adata

        for cell_type in cell_types_to_remove:
            modified = modified[modified.obs[obs_label_column] != cell_type]
        
        return modified

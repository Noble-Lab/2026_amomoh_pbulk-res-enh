"""
hic_dataset.py

PyTorch Dataset for the paired input/target submatrix HDF5 files built by
scripts/build_submatrix_dataset.py.  Create a separate instance for each
split, for example, SubmatrixDataset("train.h5") for training,
SubmatrixDataset("val.h5") for validation, and SubmatrixDataset("test.h5")
for testing.
"""


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class SubmatrixDataset(Dataset):
    def __init__(self, h5_path):
        # Keep the file handle open for the dataset's lifetime rather than
        # reopening per __getitem__ call, h5py supports random-access reads
        # without loading the whole file into memory.
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, "r") 
        self.input = self.file["input"]
        self.target = self.file["target"]
        self.chroms = self.file["chrom"].asstr()[:]  # decode once, used by plot_prediction_comparison()

        n_input = self.input.shape[0]
        n_target = self.target.shape[0]

        if n_input != n_target:
            raise ValueError(f"{h5_path}: input has {n_input} windows but target has {n_target}, this is a mismatched file")

        self.n = n_input


    def __len__(self):
        return self.n


    def __getitem__(self, index):
        # Add a channel dimension (1, H, W), which is the standard for image-style models,
        # even though there is only one channel (contact counts).
        x = torch.from_numpy(self.input[index][None, :, :].astype(np.float32))
        y = torch.from_numpy(self.target[index][None, :, :].astype(np.float32))

        return x, y
    

    def indices_for_chrom(self, chrom):
        # Dataset indices belonging to a given chromosome, used by
        # plot_prediction_comparison() to sample representative windows.
        return np.flatnonzero(self.chroms == chrom).tolist()


    def close(self):
        self.file.close()
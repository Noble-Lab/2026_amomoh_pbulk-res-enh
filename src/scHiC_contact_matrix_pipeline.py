"""
scHiC_contact_matrix_pipeline.py

Core library functions for building and visualizing single-cell Hi-C
contact matrices from .allValidPairs.txt files. 
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def _parse_header(pairs_file):
    # Parse the leading '#' header block in a single pass, stopping at the
    # first non-header line since headers are always contiguous at the top
    chrom_sizes = {}
    columns = None
    with open(pairs_file) as file:
        for line in file:
            if not line.startswith("#"):
                break

            line = line.strip()

            # Handles both header variants seen in real GEO files:
            # '#chromsize: chrN size' and '#chromosome: chrN size'
            if line.startswith("#chromsize:") or line.startswith("#chromosome:"):
                _, chrom, size = line.split()
                chrom_sizes[chrom] = int(size)

            # Some cell types (e.g. GM12878) include extra columns
            # (phase0, phase1) beyond the standard readID/chr1/pos1/chr2/pos2/strand1/strand2 set
            elif line.startswith("#columns:"):
                columns = line.split(":", 1)[1].split()

    # Fallback for files with no '#columns:' header
    if columns is None:
        columns = ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"]

    return chrom_sizes, columns


def build_contact_matrix(pairs_file, bin_size = 1000000):
    # Step 1: Parse header once: chromosome sizes and true column layout
    chrom_sizes, columns = _parse_header(pairs_file)

    # Step 2: Load contact pairs, skipping '#' header lines
    # (handles extra trailing columns like phase0/phase1 present in some cell types)
    df = pd.read_csv(pairs_file, sep = '\t', comment = '#', header = None)
    df.columns = columns

    # Keep only intra-chromosomal contacts (both ends on the same chromosome)
    df = df[df["chr1"] == df["chr2"]]

    num, non_num = [], []
    
    for chrom in chrom_sizes.keys():
        chrom_num = chrom.replace("chr", "")
        if chrom_num.isdigit():
            num.append((int(chrom_num), chrom))
        
        else:
            non_num.append(chrom)

    chrom_sorted = [chrom for _, chrom in sorted(num)] + sorted(non_num)

    # chrom_offsets[chrom] = index of that chromosome's first bin in the global matrix
    chrom_offsets = {}
    offset = 0
    for chrom in chrom_sorted:
        chrom_offsets[chrom] = offset
        offset += int(np.ceil(chrom_sizes[chrom] / bin_size))

    total_bins = offset

    # Step 3: Filter, then assign global bin indices
    # Global bin: chromosome's starting offset + (position // bin_size)
    df = df[df["chr1"].isin(chrom_offsets)]
    bin_i = df["chr1"].map(chrom_offsets) + (df["pos1"] // bin_size)
    bin_j = df["chr2"].map(chrom_offsets) + (df["pos2"] // bin_size)

    # Step 4: Build a symmetric sparse matrix
    data = np.ones(len(bin_i))
    matrix = sp.coo_matrix((data, (bin_i, bin_j)), shape = (total_bins, total_bins)).tocsr()

    # Add the transpose to make the matrix symmetric, then halve the diagonal to correct for double-counting
    matrix = matrix + matrix.T
    diagonal = sp.diags(matrix.diagonal() / 2)
    matrix = matrix - diagonal

    return matrix, chrom_offsets, chrom_sizes


def plot_contact_matrix(built_matrix, chrom, chrom_offsets, chrom_sizes, bin_size = 1000000):
    # Extract the submatrix for this chromosome
    start = chrom_offsets[chrom]
    n_bins = int(np.ceil(chrom_sizes[chrom] / bin_size))
    end = start + n_bins

    sub_matrix = built_matrix[start:end, start:end].toarray()

    # Log-transform so the diagonal doesn't drown out everything else
    sub_matrix = np.log1p(sub_matrix)

    fig, ax = plt.subplots(figsize = (10, 6))
    im = ax.imshow(sub_matrix, cmap = "Reds", aspect = "equal")
    ax.set_adjustable("box")
    ax.set_title(f"Contact Matrix: {chrom} ({bin_size // 1000}kb bins)", fontsize = 12, fontweight = "bold")
    ax.set_xlabel(f"{chrom} ({bin_size // 1000} kb bins)", fontweight = "bold")
    ax.set_ylabel(f"{chrom} ({bin_size // 1000} kb bins)", fontweight = "bold")
    plt.colorbar(im, ax = ax, label = "log(1 + contacts)")
    plt.tight_layout()
    plt.show()
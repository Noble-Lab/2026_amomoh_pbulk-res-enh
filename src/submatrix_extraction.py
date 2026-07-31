"""
submatrix_extraction.py

Builds the train/validation/test submatrix datasets for the resolution- enhancement model, 
for the confirmed design:

Train:      input = A fixed random 10-40% cell subsample (own random draw
            per celltype) of Ex_L3/4_IT + Inh_MSN, pooled once.

            target = 100% of cells (pooled) from Ex_L3/4_IT + Inh_MSN,
            and uses all chromosomes excluding chr4, chr5, chr11, and chr14

Validation: SAME fixed input/target pools as train (same celltypes),
            and uses chr4 and chr11 only.

Test:       input = A fixed random 10-40% cell subsample of Ex_L2/3_IT
            (own independent random draw, separate from train/val's).
            
            target = 100% of cells (pooled) from Ex_L2/3_IT's,
            and uses chr5 and chr14 only

Every split uses the same input/target pattern: input is genuinely
degraded (a random cell subsample), target is always the full 100%-cell
pool. 

Test measures whether the model's enhancement ability generalizes
to a celltype it never trained on, same task as train/val, held-out data.

No chromosome is ever used in more than one split, for any cell: 
chr4/11 never appear in train and chr5/14 never appear in train or val.

Submatrices are window x window bin squares taken on the diagonal of the
pooled pseudo-bulk contact matrix, using a stride of 1 for each chromosome.
"""

import json
import glob
import os
import numpy as np
import scipy.sparse as sp
from scHiC_pbulk_batch import safe_dirname


def load_split_config(config_path):
    # Loads configs/submatrix_split.json. Required keys: train_val_cell_types,
    # test_cell_type, val_chroms, test_chroms, bin_size, stride, window.
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Split config not found: {config_path}")

    with open(config_path) as file:
        try:
            config = json.load(file)

        except json.JSONDecodeError as error:
            raise ValueError(f"Split config {config_path} is not valid JSON: {error}") from error
 
    required_keys = {"train_val_cell_types", "test_cell_type", "val_chroms", "test_chroms",
                      "bin_size", "stride", "window"}

    missing = required_keys - config.keys()

    if missing:
        raise ValueError(f"Config {config_path} is missing required key(s): {sorted(missing)}")

    return config


def frac_label(frac_range):
    # Turns a fraction range into a folder-safe label, e.g. (0.05, 0.05) -> "frac_0.05",
    # (0.1, 0.4) -> "frac_0.1-0.4". Used to auto-name output folders so different
    # fraction runs never collide or need to be renamed by hand.
    low, high = frac_range

    if low == high:
        return f"frac_{low}"

    return f"frac_{low}-{high}"


def pool_random_fraction(matrices_dir, cell_type, frac_range = (0.1, 0.4), seed = 42):
    # Draws one random fraction in frac_range and ONE random cell subset at
    # that fraction, pools just those cells into a single low-coverage matrix. 
    # This is the "input" pool and separate it from the full 100%-cell pool used as the target.

    cell_type_dir = os.path.join(matrices_dir, safe_dirname(cell_type))
    cell_files = sorted(glob.glob(os.path.join(cell_type_dir, "*.npz")))

    if len(cell_files) < 2:
        raise ValueError(f"Need at least 2 cells to subsample, found {len(cell_files)} in {cell_type_dir}")

    rng = np.random.default_rng(seed)
    fraction = rng.uniform(*frac_range)
    n_pick = max(1, round(len(cell_files) * fraction))
    chosen = rng.choice(cell_files, size = n_pick, replace = False)

    pooled = None
    for c in chosen:
        m  = sp.load_npz(c)
        pooled = m if pooled is None else pooled + m

    print(f"{cell_type}: input pool = {n_pick} / {len(cell_files)} cells (fraction = {fraction:.3f}, seed = {seed})")


    return pooled, fraction, n_pick


def extract_diagonal_windows(matrix, chroms, chrom_offsets, chrom_sizes, bin_size, cell_type, window = 256, stride = 1):
    # Slides a window-bin square along the diagonal of matrix, for each chromosome in chroms, step stride. 
    # Yields (submatrix, metadata) so each window can be traced back to its cell type, chromosome, and position.
    for chrom in chroms:
        if chrom not in chrom_offsets:
            print(f"WARNING: {chrom} not found in chrom_offsets, SKIP")
            continue

        start_offset = chrom_offsets[chrom]
        n_bins = int(np.ceil(chrom_sizes[chrom] / bin_size))

        if n_bins < window:
            print(f"WARNING: {chrom} has only {n_bins} bins, smaller than window = {window}, SKIP")
            continue

        for local_start in range(0, n_bins - window + 1, stride):
            g_start = start_offset + local_start
            g_end = g_start + window

            sub = matrix[g_start:g_end, g_start:g_end].toarray().astype(np.float32)

            yield sub, {
                "cell_type": cell_type,
                "chrom": chrom,
                "start_bin": local_start    # bin index within the chromosome, not global
            }


def extract_paired_windows(input_matrix, target_matrix, chroms, chrom_offsets, chrom_sizes, bin_size, cell_type, window = 256, stride = 1):
    # Walks input_matrix and target_matrix side by side, asserting matching positions (guarantees no misalignment).
    input_gen = extract_diagonal_windows(input_matrix, chroms, chrom_offsets, chrom_sizes, bin_size, cell_type, window, stride)
    target_gen = extract_diagonal_windows(target_matrix, chroms, chrom_offsets, chrom_sizes, bin_size, cell_type, window, stride)

    for (in_sub, in_meta), (tar_sub, tar_meta) in zip(input_gen, target_gen):
         assert in_meta["chrom"] == tar_meta["chrom"] and in_meta["start_bin"] == tar_meta["start_bin"], (
            f"Input/target position mismatch: {in_meta} vs {tar_meta}"
        )
         yield in_sub, tar_sub, in_meta
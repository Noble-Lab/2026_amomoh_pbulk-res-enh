"""
build_submatrix_dataset.py

Builds the three HDF5 datasets (train/val/test) for the resolution-enhancement model. 
See src/submatrix_extraction.py for the full split design (chromosome-only split).
 
A random low-coverage "input" pool paired against the full-coverage
"target" pool for each window position.

Reuses the pooled pseudo-bulk .npz matrices already built by run_pbulk_pipeline.py,
for targets; builds new low-coverage pools for train/val inputs from the per-cell matrices directly.

Run: python scripts/build_submatrix_dataset.py --split-config configs/submatrix_split.json \
    --pbulk-dir results/pseudo_bulk/2026-07-15/50kb \
    --matrices-dir results/matrices/2026-07-15/50kb \
    --out-dir results/submatrices \
    --frac 0.05
"""


import argparse
import os
import sys
import h5py
import numpy as np
import scipy.sparse as sp

# Add src/ to the path so this script runs directly without installing a package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from scHiC_pbulk_batch import load_chrom_map, safe_dirname
from submatrix_extraction import load_split_config, pool_random_fraction, extract_paired_windows, frac_label

def parse_args():
    parser = argparse.ArgumentParser(description = "Build train/val/test submatrix HDF5 datasets")
    parser.add_argument("--split-config", required = True, help = "Split design config, e.g. configs/submatrix_split.json")
    parser.add_argument("--pbulk-dir", required = True,
                        help = "Pooled pseudo-bulk matrices dir for the 50kb resolution, "
                            "e.g. results/pseudo_bulk/2026-07-15/50kb")
    parser.add_argument("--matrices-dir", required = True,
                        help = "Per-cell matrices dir for the same resolution (used only to "
                            "read chrom_map.json) e.g. results/matrices/2026-07-15/50kb")
    parser.add_argument("--out-dir", default = "results/submatrices",
                        help = "Where to write train.h5 / val.h5 / test.h5")
    parser.add_argument("--frac", type = float, nargs = "+", required = True,
                        help = "Input fraction: one value for fixed (e.g. --frac 0.05), "
                            "two values for a random range (e.g. --frac 0.1 0.4)")
    parser.add_argument("--seed", type = int, default = 42,
                        help = "Seed for the random input-fraction/cell-subset draws")
    
    return parser.parse_args()


def all_train_val_chroms(chrom_sizes, excluded):
    return [c for c in chrom_sizes if c not in excluded]


def load_pooled_matrix(pbulk_dir, cell_type):
    path = os.path.join(pbulk_dir, f"{safe_dirname(cell_type)}_pbulk.npz")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No pooled pbulk matrix found for {cell_type} at {path}, run run_pbulk_pipeline.py first")

    return sp.load_npz(path)


def write_h5(out_path, inputs, targets, meta, bin_size, window, stride):
    # inputs/targets: lists of (window, window) float32 arrays, same order/length, same positions.
    n = len(inputs)

    if n == 0:
        print(f"WARNING: no windows to write for {out_path}, SKIP")
        return
 
    input_arr = np.stack(inputs, axis = 0)
    target_arr = np.stack(targets, axis = 0)
    cell_types = np.array([m["cell_type"] for m in meta], dtype = h5py.string_dtype())
    chroms = np.array([m["chrom"] for m in meta], dtype = h5py.string_dtype())
    start_bins = np.array([m["start_bin"] for m in meta], dtype = np.int32)
 
    os.makedirs(os.path.dirname(out_path), exist_ok = True)

    with h5py.File(out_path, "w") as file:
        # chunks=(1, window, window): each row is its OWN compressed chunk, matching
        # how DataLoader actually reads (one full row at a time, random order).
        # Without this, h5py auto-picks a chunk shape that spans many rows and
        # splits each image into small tiles (confirmed: (125,16,16) on this data) --
        # every single random-row read then has to decompress a chunk covering 125
        # unrelated rows just to get one.

        file.create_dataset("input", data = input_arr, compression = "gzip", compression_opts = 4,
                             chunks = (1, input_arr.shape[1], input_arr.shape[2]))
        file.create_dataset("target", data = target_arr, compression = "gzip", compression_opts = 4,
                             chunks = (1, target_arr.shape[1], target_arr.shape[2]))
        file.create_dataset("cell_type", data = cell_types)
        file.create_dataset("chrom", data = chroms)
        file.create_dataset("start_bin", data = start_bins)
        file.attrs["bin_size"] = bin_size
        file.attrs["window"] = window
        file.attrs["stride"] = stride
 
    print(f"Wrote {n:,} paired windows -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
 
 
def main():
    args = parse_args()
    split = load_split_config(args.split_config)
 
    train_val_cell_types = split["train_val_cell_types"]
    test_cell_type = split["test_cell_type"]
    val_chroms = split["val_chroms"]
    test_chroms = split["test_chroms"]
    bin_size = split["bin_size"]
    stride = split["stride"]
    window = split["window"]

    if len(args.frac) == 1:
        frac_range = (args.frac[0], args.frac[0])

    elif len(args.frac) == 2:
        frac_range = (args.frac[0], args.frac[1])

    else:
        raise ValueError(f"--frac takes 1 value (fixed) or 2 values (range), got {len(args.frac)}: {args.frac}")

    out_dir = os.path.join(args.out_dir, frac_label(frac_range))
    print(f"Output folder: {out_dir}")
 
    excluded_from_train = set(val_chroms) | set(test_chroms)
 
    train_inputs, train_targets, train_meta = [], [], []
    val_inputs, val_targets, val_meta = [], [], []
    saved_bin_size = None
 
    for cell_type in train_val_cell_types:
        chrom_sizes, chrom_offsets, total_bins, cell_bin_size = load_chrom_map(args.matrices_dir, cell_type)

        if cell_bin_size != bin_size:
            raise ValueError(f"{cell_type}: matrices were built at {cell_bin_size}, config expects {bin_size}")

        saved_bin_size = cell_bin_size
 
        train_chroms = all_train_val_chroms(chrom_sizes, excluded_from_train)
 
        target_pool = load_pooled_matrix(args.pbulk_dir, cell_type)
        input_pool, fraction, n_cells = pool_random_fraction(
            args.matrices_dir, cell_type, frac_range, args.seed
        )
 
        for in_sub, tgt_sub, meta in extract_paired_windows(
            input_pool, target_pool, train_chroms, chrom_offsets, chrom_sizes, bin_size,
            cell_type, window, stride
        ):
            train_inputs.append(in_sub)
            train_targets.append(tgt_sub)
            train_meta.append(meta)
 
        for in_sub, tgt_sub, meta in extract_paired_windows(
            input_pool, target_pool, val_chroms, chrom_offsets, chrom_sizes, bin_size,
            cell_type, window, stride
        ):
            val_inputs.append(in_sub)
            val_targets.append(tgt_sub)
            val_meta.append(meta)
 
    # Test cell type: own independent random input draw, same mechanism as train/val
    chrom_sizes, chrom_offsets, total_bins, cell_bin_size = load_chrom_map(args.matrices_dir, test_cell_type)
    if cell_bin_size != bin_size:
        raise ValueError(f"{test_cell_type}: matrices were built at {cell_bin_size}, config expects {bin_size}")
 
    test_target_pool = load_pooled_matrix(args.pbulk_dir, test_cell_type)
    test_input_pool, test_fraction, test_n_cells = pool_random_fraction(
        args.matrices_dir, test_cell_type, frac_range, args.seed
    )
 
    test_inputs, test_targets, test_meta = [], [], []
    for in_sub, tgt_sub, meta in extract_paired_windows(
        test_input_pool, test_target_pool, test_chroms, chrom_offsets, chrom_sizes, bin_size,
        test_cell_type, window, stride
    ):
        test_inputs.append(in_sub)
        test_targets.append(tgt_sub)
        test_meta.append(meta)
 
    write_h5(os.path.join(out_dir, "train.h5"), train_inputs, train_targets, train_meta, bin_size, window, stride)
    write_h5(os.path.join(out_dir, "val.h5"), val_inputs, val_targets, val_meta, bin_size, window, stride)
    write_h5(os.path.join(out_dir, "test.h5"), test_inputs, test_targets, test_meta, bin_size, window, stride)
 
    print("\nDone.")
    print(f"Train: {len(train_inputs):,} paired windows")
    print(f"Val:   {len(val_inputs):,} paired windows")
    print(f"Test:  {len(test_inputs):,} paired windows")
 
 
if __name__ == "__main__":
    main()
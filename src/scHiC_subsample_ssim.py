"""
scHiC_subsample_ssim.py
 
Compares partial pseudo-bulk matrices (built from random subsets of a cell
type's cells) against the full pseudo-bulk target, using SSIM per
chromosome. Sweeps subsample size from 1 cell up to N-1 cells, with several
independent random replicates per size, to see how pseudo-bulk quality
depends on how many cells are pooled.
"""


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from skimage.metrics import structural_similarity as ssim
from scHiC_pbulk_batch import load_chrom_map, bin_size_label, safe_dirname


def list_cell_matrix_files(matrices_dir, cell_type):
    # All per-cell .npz files for this cell type (excludes chrom_map.json)
    cell_type_dir = os.path.join(matrices_dir, safe_dirname(cell_type))
    return sorted(glob.glob(os.path.join(cell_type_dir, "*.npz")))


def build_subset_pbulk(cell_files, indices):
    # Sum the matrices at the given indices into one pooled matrix
    pooled = None
    for i in indices:
        m = sp.load_npz(cell_files[i])
        pooled = m if pooled is None else pooled + m
    
    return pooled


def chrom_ssim(subset_matrix, target_matrix, chrom, chrom_offsets, chrom_sizes, bin_size):
    # SSIM between one chromosome's submatrix of subset vs target, on log1p-transformed values
    start = chrom_offsets[chrom]
    n_bins = int(np.ceil(chrom_sizes[chrom] / bin_size))
    end = start + n_bins

    sub_a = np.log1p(subset_matrix[start:end, start:end].toarray())
    sub_b = np.log1p(target_matrix[start:end, start:end].toarray())

    data_range = sub_b.max() - sub_b.min()

    if data_range == 0:
        return 1.0 # degenerate case: target chromosome slice has no signal at all
    
    return ssim(sub_a, sub_b, data_range = data_range)


def run_subsample_sweep(matrices_root, pbulk_root, cell_type, bin_size, n_replicates = 10, seed = 42):
    # Resolve the bin-size-labeled subfolder automatically, same layout run_pbulk_pipeline.py uses
    label = bin_size_label(bin_size)
    matrices_dir = os.path.join(matrices_root, label)
    pbulk_dir = os.path.join(pbulk_root, label)

    cell_files = list_cell_matrix_files(matrices_dir, cell_type)
    n_total = len(cell_files)

    if n_total < 2:
        raise ValueError(f"Need at least 2 cells to run a subsample sweep, found {n_total} in {matrices_dir}")
    
    # safe_dirname: matches how run_pbulk_pipeline.py names the pooled .npz file
    target = sp.load_npz(os.path.join(pbulk_dir, f"{safe_dirname(cell_type)}_pbulk.npz"))

    # Confirm the requested bin_size actually matches what this folder was built at,
    # rather than silently trusting the caller -- a mismatch here would otherwise
    # slice chromosomes at the wrong bin boundaries without any error.
    chrom_sizes, chrom_offsets, total_bins, saved_bin_size = load_chrom_map(matrices_dir, cell_type)
    
    if saved_bin_size != bin_size:
        raise ValueError(
            f"bin_size mismatch: requested {bin_size} but {matrices_dir} was built at {saved_bin_size}"
        )
    
    chroms = sorted(chrom_sizes.keys())

    rng = np.random.default_rng(seed)
    results = []

    # For each subset size, draw n_replicates independent random subsets,
    # pool each one, and score it against the full target per chromosome
    for n_cells in range(1, n_total + 1):  # 1 ... N, full set
        for replicate in range(n_replicates):
            indices = rng.choice(n_total, size = n_cells, replace = False)
            subset = build_subset_pbulk(cell_files, indices)

            scores = [
                chrom_ssim(subset, target, chrom, chrom_offsets, chrom_sizes, bin_size)

                for chrom in chroms
            ]

            results.append({
                "cell_type": cell_type,
                "bin_size": bin_size,
                "n_cells": n_cells,
                "n_total": n_total,
                "replicate": replicate,
                "mean_ssim": float(np.mean(scores)),
                "median_ssim": float(np.median(scores))
            })
    
    
    return results

def plot_subsample_curve(results, title = None):
    # x-axis: fraction of the full pseudo-bulk (n_cells / n_total)
    # y-axis: SSIM, as two separate plots -- mean-of-mean and mean-of-median,
    # each averaged across the replicates drawn at that subset size
    n_total = results[0]["n_total"]

    by_n = {}
    for r in results:
        by_n.setdefault(r["n_cells"], {"mean": [], "median": []})
        by_n[r["n_cells"]]["mean"].append(r["mean_ssim"])
        by_n[r["n_cells"]]["median"].append(r["median_ssim"])

    n_cells_sorted = sorted(by_n.keys())
    fractions = [n / n_total for n in n_cells_sorted]
    mean_line = [np.mean(by_n[n]["mean"]) for n in n_cells_sorted]
    median_line = [np.mean(by_n[n]["median"]) for n in n_cells_sorted]

    base_title = title or "Pseudo-bulk SSIM vs Subset Fraction"

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(fractions, mean_line, color = "#E66101", linestyle = "-", linewidth = 2.5, marker = "o", label = "Mean SSIM")
    ax.plot(fractions, median_line, color = "#5E3C99", linestyle = "--", linewidth = 2.5, marker = "s", label = "Median SSIM")
    ax.set_xlabel("Fraction of Pseudo-bulk", fontweight = "bold")
    ax.set_ylabel("SSIM Score", fontweight = "bold")
    ax.set_title(base_title, fontweight = "bold")
    ax.legend()
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.show()


def run_all_subsample_sweeps(matrices_root, pbulk_root, bin_sizes, cell_types, n_replicates = 10, seed = 42):
    # Runs run_subsample_sweep for every (cell_type, bin_size) combination and plotting
    # each one. Returns all raw results keyed by (cell_type, bin_size),
    # so the data is still reachable even when plotting is on.
    # cell_types has no default now, pass config["cell_type_dirs"].keys() from the
    # relevant dataset config (see configs/), same as run_pbulk_pipeline.py does.

    all_results = {}

    for cell_type in cell_types:

        for bin_size in bin_sizes:
            label = bin_size_label(bin_size)
            print(f"---- {cell_type} @ {label} ----")
 
            results = run_subsample_sweep(
                matrices_root, pbulk_root, cell_type, bin_size,
                n_replicates, seed
            )
            all_results[(cell_type, bin_size)] = results

            plot_subsample_curve(results, title = f"{cell_type} Cell SSIM vs Subset Fraction of Pseudo-bulk ({label})")

    return all_results

"""
scHiC_pbulk_batch.py
 
Loops over per-cell-type pairs files, builds + saves per-cell
contact matrices, and pools them into pseudo-bulk matrices. Depends on
build_contact_matrix from scHiC_contact_matrix_pipeline.py.

Dataset-specific values (cell type -> directory mapping, pairs file
suffix) are NOT hardcoded here, they're loaded from a per-dataset
config file (see configs/*.json) and passed in by the caller
(run_pbulk_pipeline.py). This keeps this file identical across every
dataset it's ever pointed at.
"""


import json
import os
import glob
import scipy.sparse as sp
from scHiC_contact_matrix_pipeline import build_contact_matrix, build_canonical_chrom_map

def load_dataset_config(config_path):
    # Loads configs/*.json. Required keys: dataset_name, pairs_suffix, and cell_type_dirs.
    # genome_build is a reference label only; no code path reads or checks it.
     
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    with open(config_path) as file:
        try:
            config = json.load(file)
        
        except json.JSONDecodeError as error:
            raise ValueError(f"Dataset config {config_path} is not valid JSON: {error}") from error
 
    required_keys = {"dataset_name", "pairs_suffix", "cell_type_dirs", "bin_sizes"}
    missing = required_keys - config.keys()    

    if missing:
        raise ValueError(f"Config {config_path} is missing required key(s): {sorted(missing)}")
 
    if not config["cell_type_dirs"]:
        raise ValueError(f"Config {config_path} has an empty cell_type_dirs, nothing to process")
 
    return config        


def safe_dirname(cell_type):
    # Cell-type labels can contain '/' (e.g. "Ex_L3/4_IT"), which the OS
    # reads as a path separator, not a literal character. Anywhere a cell
    # type name is used to build a FILE or DIRECTORY name (as opposed to
    # looking something up in a dict), it needs to go through this first.
    return cell_type.replace("/", "_") 


def bin_size_label(bin_size):
    # 1000000 -> "1mb", 500000 -> "500kb", 250000 -> "250kb"
    if bin_size % 1000000 == 0:
        return f"{bin_size // 1000000}mb"
    
    return f"{bin_size // 1000}kb"


def process_all_cells(data_dir, output_dir, cell_type_dirs, pairs_suffix, bin_size = 1000000):
    # Loop every cell type's pairs files, build a matrix for each cell,
    # and save as .npz under output_dir/<safe cell_type>/<cell_name>.npz
    # cell_type_dirs / pairs_suffix come from a dataset config (see configs/)
    os.makedirs(output_dir, exist_ok = True)

    for cell_type, subdir in cell_type_dirs.items():
        cell_type_data_dir = os.path.join(data_dir, subdir)
        cell_type_out_dir = os.path.join(output_dir, safe_dirname(cell_type))
        os.makedirs(cell_type_out_dir, exist_ok = True)

        pairs_files = sorted(glob.glob(os.path.join(cell_type_data_dir, f"*{pairs_suffix}")))

        if not pairs_files:
            print(f"WARNING: no *{pairs_suffix} files found for {cell_type} in {cell_type_data_dir}")
            continue

        print(f"{cell_type}: {len(pairs_files)} cells found")

        # One shared chrom map per cell type -- keeps every cell's matrix the same
        # shape even if individual headers list slightly different chromosomes
        # (e.g. a cell with zero chrY contacts omitting chrY from its header).
        chrom_sizes, chrom_offsets, total_bins = build_canonical_chrom_map(pairs_files, bin_size)

        # Save it alongside the matrices so plotting/inspection later doesn't need
        # to rescan every raw .allValidPairs.txt file just to get this back.
        chrom_map_path = os.path.join(cell_type_out_dir, "chrom_map.json")
        
        with open(chrom_map_path, "w") as file:
            json.dump({
                "bin_size": bin_size,
                "chrom_sizes": chrom_sizes,
                "chrom_offsets": chrom_offsets,
                "total_bins": total_bins,
            }, file, indent = 2)
        
        for pairs_file in pairs_files:
            cell_name = os.path.basename(pairs_file).replace(pairs_suffix, "")
            out_path = os.path.join(cell_type_out_dir, f"{cell_name}.npz")            
            
            # Skip cells already processed so a run can be resumed without redoing work
            if os.path.exists(out_path):
                print(f"  Skipping {cell_name} (already processed)")
                continue
 
            print(f"  Processing {cell_name}...")
            matrix, _, _ = build_contact_matrix(
                pairs_file, bin_size,
                chrom_sizes = chrom_sizes, chrom_offsets = chrom_offsets, total_bins = total_bins
            )
            sp.save_npz(out_path, matrix)
 
    print("Done.")


def load_chrom_map(output_dir, cell_type):
    # Load the chrom map process_all_cells already saved for this cell type,
    # instead of rescanning every raw file again (e.g. when plotting a pooled
    # matrix in the notebook).
    chrom_map_path = os.path.join(output_dir, safe_dirname(cell_type), "chrom_map.json")
    with open(chrom_map_path) as file:
        chrom_map = json.load(file)
 
    return chrom_map["chrom_sizes"], chrom_map["chrom_offsets"], chrom_map["total_bins"], chrom_map["bin_size"]


def pseudo_bulk(output_dir, cell_type):
    # Sum every per-cell matrix for a cell type into one pooled matrix
    cell_type_dir = os.path.join(output_dir, safe_dirname(cell_type))
    files = sorted(glob.glob(os.path.join(cell_type_dir, "*.npz")))
 
    if not files:
        raise FileNotFoundError(f"No .npz matrices found for {cell_type} in {cell_type_dir}")
 
    pooled = None
    n_cells = 0
    
    for file in files:
        m = sp.load_npz(file)
        pooled = m if pooled is None else pooled + m
        n_cells += 1
 
    print(f"{cell_type}: pooled {n_cells} cells")
    return pooled
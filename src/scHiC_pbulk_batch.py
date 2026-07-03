"""
scHiC_pbulk_batch.py
 
Loops over per-cell-type .allValidPairs.txt files, builds + saves per-cell
contact matrices, and pools them into pseudo-bulk matrices. Depends on
build_contact_matrix from scHiC_contact_matrix_pipeline.py.
"""


import os
import glob
import scipy.sparse as sp
from scHiC_contact_matrix_pipeline import build_contact_matrix


# Maps a cell type name to its actual data directory
# eHAP uses the "standard" subfolder, "e_variant" files (.e.allValidPairs.txt) are excluded.
CELL_TYPE_DIRS = {
    "BJ": "BJ",
    "K562": "K562",
    "eHAP": os.path.join("eHAP", "standard"),
    "GM12878": "GM12878"
}


def process_all_cells(data_dir, output_dir, cell_type_dirs = CELL_TYPE_DIRS, bin_size = 1000000):
    # Loop every cell type's .allValidPairs.txt files, build a matrix for each cell,
    # and save as .npz under output_dir/<cell_type>/<cell_name>.npz
    os.makedirs(output_dir, exist_ok = True)

    for cell_type, subdir in cell_type_dirs.items():
        cell_type_data_dir = os.path.join(data_dir, subdir)
        cell_type_out_dir = os.path.join(output_dir, cell_type)
        os.makedirs(cell_type_out_dir, exist_ok = True)

        pairs_files = sorted(glob.glob(os.path.join(cell_type_data_dir, "*.allValidPairs.txt")))

        if not pairs_files:
            print(f"WARNING: no .allValidPairs.txt files found for {cell_type} in {cell_type_data_dir}")
            continue

        print(f"{cell_type}: {len(pairs_files)} cells found")

        for pairs_file in pairs_files:
            cell_name = os.path.basename(pairs_file).replace(".allValidPairs.txt", "")
            out_path = os.path.join(cell_type_out_dir, f"{cell_name}.npz")            
            
            # Skip cells already processed so a run can be resumed without redoing work
            if os.path.exists(out_path):
                print(f"  Skipping {cell_name} (already processed)")
                continue
 
            print(f"  Processing {cell_name}...")
            matrix, chrom_offsets, chrom_sizes = build_contact_matrix(pairs_file, bin_size)
            sp.save_npz(out_path, matrix)
 
    print("Done.")


def pseudo_bulk(output_dir, cell_type):
    # Sum every per-cell matrix for a cell type into one pooled matrix
    cell_type_dir = os.path.join(output_dir, cell_type)
    files = sorted(glob.glob(os.path.join(cell_type_dir, "*.npz")))
 
    if not files:
        raise FileNotFoundError(f"No .npz matrices found for {cell_type} in {cell_type_dir}")
 
    pooled = None
    n_cells = 0
    for f in files:
        m = sp.load_npz(f)
        pooled = m if pooled is None else pooled + m
        n_cells += 1
 
    print(f"{cell_type}: pooled {n_cells} cells")
    return pooled
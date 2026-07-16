"""
run_pbulk_pipeline.py
 
Calls process_all_cells and pseudo_bulk from src/ and runs them across the full dataset
 
Dataset-specific values (cell type -> directory mapping, pairs file suffix) come from
a per-dataset config file (see configs/*.json), not from hardcoded constants, this lets
the same src/ code serve multiple datasets without duplication.
 
Run: python scripts/run_pbulk_pipeline.py --config configs/GSE303006.json --data-dir data/2026-07-15 --output-dir results/matrices
"""


import argparse
import os
import sys
import scipy.sparse as sp

# Add src/ to the path so this script runs directly without installing a package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from scHiC_pbulk_batch import process_all_cells, pseudo_bulk, bin_size_label, safe_dirname, load_dataset_config

# Resolutions to build automatically on every run
BIN_SIZES = [1000000, 500000, 250000]


def parse_args():
    parser = argparse.ArgumentParser(description = "Run the scHi-C pseudo-bulk pipeline")
    parser.add_argument("--config", required = True,
                        help = "Path to a per-dataset config file, e.g. configs/GSE303006.json")
    parser.add_argument("--data-dir", default = "data/2026-06-29",
                        help = "Root directory containing per-cell-type pairs-file folders")
    parser.add_argument("--output-dir", default = "results/matrices",
                        help = "Directory to write per-cell .npz contact matrices")
    parser.add_argument("--pbulk-dir", default = "results/pseudo_bulk",
                        help = "Directory to write pooled pseudo-bulk .npz matrices")
    parser.add_argument("--cell-types", nargs = "+", default = None,
                        help = "Subset of cell types to process (default: all in the config)")
    parser.add_argument("--skip-pbulk", action = "store_true",
                        help = "Skip pseudo-bulk pooling, only build per-cell matrices")

    return parser.parse_args()    


def run_for_bin_size(args, bin_size, cell_type_dirs, pairs_suffix):
    # Nest a bin_size subfolder into both output dirs so different resolutions
    # (1mb, 500kb, 250kb, etc.) land in separate folders automatically
    label = bin_size_label(bin_size)
    output_dir = os.path.join(args.output_dir, label)
    pbulk_dir = os.path.join(args.pbulk_dir, label)

    print(f"Building per-cell matrices from {args.data_dir} -> {output_dir}")

    process_all_cells(
        data_dir = args.data_dir,
        output_dir = output_dir,
        cell_type_dirs = cell_type_dirs,
        pairs_suffix = pairs_suffix,
        bin_size = bin_size
    )

    if not args.skip_pbulk:
        os.makedirs(pbulk_dir, exist_ok = True)
        print(f"\nPooling pseudo-bulk matrices -> {pbulk_dir}")

        for cell_type in cell_type_dirs:
            
            try:
                pooled = pseudo_bulk(output_dir, cell_type)

            # No matrices built yet for this cell type (e.g. empty data folder) -- skip it, don't crash the whole run
            except FileNotFoundError as error:
                print(f"  {cell_type}: {error}")
                continue

            # safe_dirname: some cell type labels contain '/' (e.g. "Ex_L3/4_IT"),
            # which can't be used directly in a filename, see src/scHiC_pbulk_batch.py
            out_path = os.path.join(pbulk_dir, f"{safe_dirname(cell_type)}_pbulk.npz")
            sp.save_npz(out_path, pooled)
            print(f"  {cell_type}: saved pooled matrix to {out_path}")
 


def main():
    args = parse_args()

    config = load_dataset_config(args.config)
    all_cell_type_dirs = config["cell_type_dirs"]
    pairs_suffix = config["pairs_suffix"]
    
    # Only keep cell types the user actually asked for and that exist in this dataset's config
    requested = args.cell_types or list(all_cell_type_dirs.keys())
    cell_type_dirs = {ct: all_cell_type_dirs[ct] for ct in requested if ct in all_cell_type_dirs}
    missing = set(requested) - set(all_cell_type_dirs)

    if missing:
        print(f"WARNING: unknown cell type(s) requested, skipping: {sorted(missing)}")

    print(f"Dataset: {config['dataset_name']} (genome build: {config.get('genome_build', 'unspecified')})")
    
    for bin_size in BIN_SIZES:
        print(f"---- bin size: {bin_size_label(bin_size)} ----")
        run_for_bin_size(args, bin_size, cell_type_dirs, pairs_suffix)
        print()
        
    
    print("\nPipeline run complete.")
 
 
if __name__ == "__main__":
    main() 
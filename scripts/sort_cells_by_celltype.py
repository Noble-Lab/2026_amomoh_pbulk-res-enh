"""
sort_cells_by_celltype.py
 
One-time setup script: reads a CHARM-style metadata table (cellname -> celltype)
and organizes a flat folder of raw .pairs files into the per-celltype folder
structure scHiC_pbulk_batch.py expects (data_dir/<safe celltype name>/*.pairs),
matching a given dataset config exactly.
 
Tries symlinks first (no extra disk space); falls back to copying if the OS
denies symlink creation (common on Windows without admin/Developer Mode).
 
Run: python scripts/sort_cells_by_celltype.py \
    --metadata data/2026-07-15/GSE303006_charm_metadata_qcpass.tsv \
    --pairs-dir data/2026-07-15/geo_submit/brain_pairs/pairs \
    --data-dir data/2026-07-15 \
    --config configs/GSE303006.json
"""


import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

# Add src/ to the path so this script runs directly without installing a package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from scHiC_pbulk_batch import load_dataset_config, safe_dirname


def parse_args():
    parser = argparse.ArgumentParser(description = "Organize raw pairs files into per-celltype folders")
    parser.add_argument("--metadata", required = True,
                        help = "Path to the cellname->celltype metadata TSV (e.g. charm_metadata_qcpass.tsv)")
    parser.add_argument("--pairs-dir", required = True,
                        help = "Folder containing the flat pile of raw .pairs files")
    parser.add_argument("--data-dir", required = True,
                        help = "Dataset root to organize into (e.g. data/2026-07-15) -- "
                               "must match the --data-dir you'll later pass to run_pbulk_pipeline.py")
    parser.add_argument("--config", required = True,
                        help = "Dataset config (e.g. configs/GSE303006.json) -- used to get "
                               "pairs_suffix and to confirm celltype folder names match exactly")
    parser.add_argument("--cellname-col", default = "cellname",
                        help = "Metadata column holding the cell identifier")
    parser.add_argument("--celltype-col", default = "celltype",
                        help = "Metadata column holding the cell type label")
    parser.add_argument("--mode", choices = ["symlink", "copy"], default = "symlink",
                        help = "symlink (default, no extra disk space) or copy (always works, costs disk space)")
 
    return parser.parse_args()

def read_cellname_to_celltype(metadata_path, cellname_col, celltype_col):
    mapping = {}

    with open(metadata_path, newline = "") as file:
        reader = csv.DictReader(file, delimiter = "\t")

        if cellname_col not in reader.fieldnames or celltype_col not in reader.fieldnames:
            raise ValueError(
                f"Expected columns '{cellname_col}' and '{celltype_col}' in {metadata_path}, "
                f"found: {reader.fieldnames}"                
            )
        
        for row in reader:
            mapping[row[cellname_col]] = row[celltype_col]

    return mapping


def link_or_copy(src, dest, mode):
    # Try the requested mode; if symlinking is denied by the OS (common on
    # Windows without admin/Developer Mode), fall back to copying instead
    # of failing the whole run over a permissions quirk.
    if mode == "symlink":

        try:
            os.symlink(src.resolve(), dest)
            return "symlink"
        
        except OSError:
            shutil.copy2(src, dest)
            return "copy (symlink denied)"
 
    shutil.copy2(src, dest)
    return "copy"


def main():
    args = parse_args()

    config = load_dataset_config(args.config)
    pairs_suffix = config["pairs_suffix"]
    known_celltypes = set(config["cell_type_dirs"].keys())
 
    pairs_dir = Path(args.pairs_dir)
    data_dir = Path(args.data_dir)
 
    cellname_to_celltype = read_cellname_to_celltype(args.metadata, args.cellname_col, args.celltype_col)
    print(f"Loaded {len(cellname_to_celltype)} cellname -> celltype mappings from {args.metadata}")
 
    linked = {}       # celltype -> count actually organized
    missing_file = []  # cells in metadata with no matching .pairs file on disk
    unknown_celltype = []  # cells whose celltype isn't in the config's cell_type_dirs
    used_copy_fallback = False
 
    for cellname, celltype in cellname_to_celltype.items():
        src = pairs_dir / f"{cellname}{pairs_suffix}"

        if not src.exists():
            missing_file.append(cellname)
            continue
 
        if celltype not in known_celltypes:
            unknown_celltype.append((cellname, celltype))
            continue
 
        dest_dir = data_dir / safe_dirname(celltype)
        dest_dir.mkdir(parents = True, exist_ok = True)
        dest = dest_dir / f"{cellname}{pairs_suffix}"
 
        if dest.exists():
            linked[celltype] = linked.get(celltype, 0) + 1  # already done, still count it
            continue
 
        result = link_or_copy(src, dest, args.mode)
        if "copy" in result and args.mode == "symlink":
            used_copy_fallback = True
 
        linked[celltype] = linked.get(celltype, 0) + 1
 
    print("\n---- summary ----")
    for celltype in sorted(linked):
        print(f"  {celltype}: {linked[celltype]} cells organized")
 
    if missing_file:
        print(f"\nWARNING: {len(missing_file)} cell(s) in metadata had no matching "
              f"*{pairs_suffix} file in {pairs_dir} (e.g. {missing_file[:3]})")
        
    if unknown_celltype:
        distinct = sorted(set(ct for _, ct in unknown_celltype))
        print(f"\nWARNING: {len(unknown_celltype)} cell(s) had a celltype not listed in "
              f"{args.config}'s cell_type_dirs: {distinct}")
 
    if used_copy_fallback:
        print("\nNOTE: symlinking was denied by the OS for at least one file (likely Windows without "
              "admin/Developer Mode) -- those files were copied instead. Enable Developer Mode "
              "(Settings > Update & Security > For Developers) to symlink and save disk space next time.")
 
    print("\nDone.")

if __name__ == "__main__":
    main() 
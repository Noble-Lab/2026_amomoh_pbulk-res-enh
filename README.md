# Enhancing scHi-C pseudobulk matrices for fine-resolution analysis of rare cell types

We repurpose a **bulk Hi-C resolution-enhancement model** ([HiCFoundation](https://github.com/Noble-Lab/HiCFoundation))
for **single-cell Hi-C (scHi-C) pseudobulk** data. The goal is to recover the fine-scale
chromatin structure (loops, TADs) that is missing from the sparse pseudobulk contact
matrices of *rare* cell types, so they can be studied at the same resolution as abundant
ones.

REU project, Department of Genome Sciences, University of Washington.
Amir Momoh, Ghulam Murtaza, William Stafford Noble.

---

## Table of contents

- [Background and approach](#background-and-approach)
- [Repository layout](#repository-layout)
- [Environment setup](#environment-setup)
- [Data acquisition](#data-acquisition)
- [Reproduction pipeline](#reproduction-pipeline)
- [Configuration files](#configuration-files)
- [Models](#models)
- [Evaluation](#evaluation)
- [Directory and naming conventions](#directory-and-naming-conventions)
- [Known caveats](#known-caveats)
- [Authors and citation](#authors-and-citation)

---

## Background and approach

Conventional Hi-C averages over millions of cells. scHi-C resolves cell-type-specific
3D genome organization but each single-cell matrix is extremely sparse
(10^4–10^6 contacts vs. ~10^8 for bulk). Cell-type structure is therefore studied by
pooling all cells of a type into a **pseudobulk** matrix. The quality of that pseudobulk
scales with the number of cells pooled, so **rare cell types stay sparse** even after
pooling — which matters because rare populations (tumor-initiating cells, senescent
cells, transient progenitor states) are often the biologically important ones.

**Approach.** HiCFoundation is a Hi-C foundation model pretrained with a
**masked-autoencoding** objective — the model is shown Hi-C submatrices with a majority
of patches masked (75% by default) and learns by reconstructing the full submatrix — on
a large collection of bulk Hi-C experiments; its internal representations stay stable as
inputs get sparser. We:

1. Freeze the HiCFoundation encoder and attach a small trainable head
   (bridge → optional transformer decoder blocks → per-patch pixel head → symmetrize).
2. Train the head on **abundant** cell types, where we can build a dense target by pooling
   all cells and a matched sparse input by subsampling a random fraction of those cells.
3. Evaluate on a **held-out cell type** to measure whether the enhancement generalizes.

A `SimpleEnhanceCNN` (3-layer fully-convolutional net) is trained on the identical data
as a baseline.

---

## Repository layout

```
configs/                     Per-dataset + split + hyperparameter JSON (no values hardcoded in src/)
  Gse240114.json               GSE240114 (LiMCA, human, 4 cell lines)
  Gse303006.json               GSE303006 (CHARM, mouse brain cortex, 19 cell types)
  submatrix_split.json         Train/val/test cell-type roles, held-out chromosomes, window/stride
  training_hyperparameters.json  batch size, lr, epochs, patience, ...

src/                         Library code (dataset-agnostic)
  scHiC_contact_matrix_pipeline.py   .pairs / .allValidPairs.txt  ->  sparse per-cell contact matrix
  scHiC_pbulk_batch.py               batch per-cell matrix build + pool into pseudobulk
  scHiC_subsample_ssim.py            subsample sweep: pseudobulk SSIM vs. #cells pooled
  submatrix_extraction.py            paired input/target 256x256 diagonal-window extraction + split logic
  hic_dataset.py                     SubmatrixDataset (PyTorch Dataset over the paired-window HDF5 files)
  hic_model.py                       SimpleEnhanceCNN, HiCFoundationHead
  hicfoundation_standalone.py        self-contained HiCFoundation encoder + resolution-enhancement head
  hic_train.py                       train loop, early stopping, SSIM/HiCRep/GenomeDISCO, plotting

scripts/                     CLI entry points that drive src/
  sort_cells_by_celltype.py          one-time: flat .pairs pile  ->  per-cell-type folders (from metadata)
  run_pbulk_pipeline.py              per-cell matrices + pooled pseudobulk, for every bin size in the config
  build_submatrix_dataset.py         build train.h5 / val.h5 / test.h5
  train_model.py                     train SimpleEnhanceCNN
  train_hicfoundation_model.py       train HiCFoundationHead (--decoder-layers 0/1/4/8)

results/
  matrices/<date>/<binsize>/<celltype>/*.npz    per-cell sparse contact matrices + chrom_map.json
  pseudo_bulk/<date>/<binsize>/<celltype>_pbulk.npz   pooled pseudobulk matrices
  submatrices/frac_<f>/{train,val,test}.h5      paired input/target window datasets
  training/<date>/                              SimpleEnhanceCNN model.pt + history.json
  training_hicfoundation_<N>layer/<date>/       HiCFoundationHead model.pt + history.json
  score_cache/*.npz                             cached per-window metric scores (SSIM/HiCRep/GenomeDISCO)
  plots/                                        generated figures
  notebook.ipynb                               running lab notebook (chronological)
  analysis/{ssim,hicrep,gdisco}_analysis.ipynb  Input vs CNN vs HiCFoundation-{1,4,8}layer comparison

data/                        Raw pairs files + dataset README (raw data is git-ignored — see data/README)
doc/                         LaTeX manuscript (pbulk-res-enh.tex) + makefile
hicfoundation_model/         Pretrained checkpoint hicfoundation_pretrain.pth.tar (~1.3 GB, git-ignored)
```

Almost everything under `results/` and all of `data/`, `hicfoundation_model/`, and
`venv310/` is git-ignored; the repo tracks only code, configs, notebooks, and docs.
See `.gitignore`.

---

## Environment setup

Developed with **Python 3.10** on Windows 11 with a CUDA 12.1 GPU. CPU-only works but
HiCFoundation training will be slow.

```bash
python3.10 -m venv venv310
# Windows:  venv310\Scripts\activate
# Linux/macOS:  source venv310/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you have a CUDA GPU, install the matching PyTorch build first, e.g.:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Core versions this project was run against: `torch 2.5.1+cu121`, `numpy 2.2.6`,
`scipy 1.15.3`, `h5py 3.16.0`, `scikit-image 0.25.2`, `pandas`, `matplotlib 3.10.9`,
`tqdm 4.70.0`. (`src/hicfoundation_standalone.py` was independently verified against
newer `torch`/`numpy` but does not depend on any bleeding-edge API.)

---

## Data acquisition

None of the raw data or the pretrained weights are tracked in git. `data/README` has the
authoritative provenance notes; a summary follows.

### GSE303006 — mouse brain cortex (used for training)

- **Source:** [GSE303006](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE303006),
  CHARM dataset, mouse brain cortex only (the mESC portion is not used). Genome GRCm38/mm10.
- **Files:** the `brain_pairs` supplementary `*.pairs` files, plus
  `GSE303006_charm_metadata_qcpass.tsv` (cell name → cell type).
- **Layout expected by the pipeline:**
  ```
  data/2026-07-15/GSE303006_charm_metadata_qcpass.tsv
  data/2026-07-15/geo_submit/brain_pairs/pairs/        *.pairs   (flat, as downloaded)
  data/2026-07-15/<celltype>/                          *.pairs   (created by sort_cells_by_celltype.py)
  ```
- 19 cortical cell types, 4265 cells total. Counts range from Ex_L3/4_IT (538 cells) down
  to Ex_Unknown2 (27 cells); see `data/README` for the full table.

### GSE240114 — human cell lines (pseudobulk sweep only)

- **Source:** [GSE240114](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240114),
  from the LiMCA paper (Wu et al. 2024, *Nat Methods*, doi:10.1038/s41592-024-02239-0).
  Genome GRCh38. ~23.5 GB raw.
- **Files:** `*.allValidPairs.txt` per cell.
- **Layout:**
  ```
  data/2026-06-29/{BJ, K562, GM12878}/     *.allValidPairs.txt
  data/2026-06-29/eHAP/standard/           *.allValidPairs.txt   (used)
  data/2026-06-29/eHAP/e_variant/          *.e.allValidPairs.txt
  ```
- Cell counts: BJ 63, K562 63, eHAP 42, GM12878 220.

### HiCFoundation pretrained weights

Download `hicfoundation_pretrain.pth.tar` from the HiCFoundation model release on
Hugging Face ([wang3702/hicfoundation_models](https://huggingface.co/wang3702/hicfoundation_models/tree/main),
[direct file](https://huggingface.co/wang3702/hicfoundation_models/blob/main/hicfoundation_pretrain.pth.tar))
and place it at:

```
hicfoundation_model/hicfoundation_pretrain.pth.tar
```

```bash
pip install huggingface_hub
huggingface-cli download wang3702/hicfoundation_models hicfoundation_pretrain.pth.tar \
  --local-dir hicfoundation_model
```

(~1.3 GB.) See the [HiCFoundation repo](https://github.com/Noble-Lab/HiCFoundation) for
model details. Only the encoder-subset keys are loaded (`strict=False`); decoder /
mask-token / head keys in the checkpoint are ignored, and `pos_embed` is regenerated from
a sin-cos grid.

---

## Reproduction pipeline

All commands are run from the repo root with the venv active. `\` line continuations are
bash; on PowerShell put each command on one line or use a backtick `` ` ``.

### 0. (GSE303006 only) sort raw pairs into per-cell-type folders

```bash
python scripts/sort_cells_by_celltype.py \
  --metadata data/2026-07-15/GSE303006_charm_metadata_qcpass.tsv \
  --pairs-dir data/2026-07-15/geo_submit/brain_pairs/pairs \
  --data-dir  data/2026-07-15 \
  --config    configs/Gse303006.json
```

Tries symlinks, falls back to copying if the OS denies them (Windows without Developer
Mode). Cell types with `/` in the name (e.g. `Ex_L3/4_IT`) become `Ex_L3_4_IT` on disk.

### 1. Build per-cell contact matrices + pooled pseudobulk

```bash
python scripts/run_pbulk_pipeline.py \
  --config     configs/Gse303006.json \
  --data-dir   data/2026-07-15 \
  --output-dir results/matrices/2026-07-15 \
  --pbulk-dir  results/pseudo_bulk/2026-07-15
```

Runs once per bin size listed in the config (`Gse303006.json`: 250 kb, 100 kb, 50 kb;
`Gse240114.json`: 1 mb, 500 kb, 250 kb), writing to a `<binsize>/` subfolder of each
output dir. Only intra-chromosomal contacts on chr1–chr22 + chrX are kept; matrices are
symmetrized with a halved diagonal. A `chrom_map.json` is saved next to each cell type's
matrices. Re-running skips cells that already have an `.npz`.

Use `--cell-types A B C` to restrict, `--skip-pbulk` to build per-cell matrices only.

### 2. (optional) Pseudobulk subsample sweep

Motivational analysis — how pseudobulk quality (SSIM vs. the full pool) degrades as fewer
cells are pooled. Run the first cells of `results/notebook.ipynb`, which call
`run_all_subsample_sweeps(...)` from `src/scHiC_subsample_ssim.py`.

### 3. Build the paired train/val/test window datasets

```bash
python scripts/build_submatrix_dataset.py \
  --split-config configs/submatrix_split.json \
  --pbulk-dir    results/pseudo_bulk/2026-07-15/50kb \
  --matrices-dir results/matrices/2026-07-15/50kb \
  --out-dir      results/submatrices \
  --frac 0.05
```

`--frac 0.05` = fixed 5 % cell subsample for the input pool; `--frac 0.1 0.4` = a random
fraction drawn once in [0.1, 0.4]. Output folder is auto-named `frac_0.05` /
`frac_0.1-0.4`. `--seed` (default 42) fixes the fraction and cell-subset draws.

The split (`configs/submatrix_split.json`) is **chromosome-disjoint** so no chromosome is
shared across splits:

| Split | Cell type(s)                  | Chromosomes                     | Input                    | Target        |
|-------|-------------------------------|---------------------------------|--------------------------|---------------|
| Train | Ex_L3/4_IT + Inh_MSN          | all except chr4, chr5, chr11, chr14 | pooled cell subsample | 100 %-cell pool |
| Val   | Ex_L3/4_IT + Inh_MSN (same pools) | chr4, chr11                 | same as train            | same as train |
| Test  | Ex_L2/3_IT (held out)         | chr5, chr14                     | own independent subsample | 100 %-cell pool |

Windows are 256×256-bin squares taken along the matrix diagonal at stride 1, per
chromosome. HDF5 is chunked `(1, 256, 256)` for fast random-row reads by the DataLoader.

### 4. Train

**Baseline CNN:**

```bash
python scripts/train_model.py \
  --config  configs/training_hyperparameters.json \
  --train-h5 results/submatrices/frac_0.05/train.h5 \
  --val-h5   results/submatrices/frac_0.05/val.h5 \
  --out-dir  results/training
```

**HiCFoundation head** — run once per decoder depth used in the comparison (0, 1, 4, 8):

```bash
python scripts/train_hicfoundation_model.py \
  --config  configs/training_hyperparameters.json \
  --train-h5 results/submatrices/frac_0.05/train.h5 \
  --val-h5   results/submatrices/frac_0.05/val.h5 \
  --hicfoundation-weights hicfoundation_model/hicfoundation_pretrain.pth.tar \
  --decoder-layers 8 \
  --out-dir  results/training_hicfoundation_8layer
```

Both scripts write `model.pt` (best val loss, restored before saving) and `history.json`
to a **dated** subfolder (`<out-dir>/YYYY-MM-DD/`). Training uses Adam, MSE loss, and
early stopping (`patience` epochs without val improvement; `train_hicfoundation_model.py`
takes a `--patience` override). The HiCFoundation encoder stays frozen and in eval mode;
only the bridge / decoder blocks / pixel head are optimized.

### 5. Evaluate and plot

Open the notebooks in `results/analysis/`:

- `ssim_analysis.ipynb` — SSIM (skimage single-scale, on `log1p` values)
- `hicrep_analysis.ipynb` — HiCRep stratum-adjusted correlation (SCC)
- `gdisco_analysis.ipynb` — GenomeDISCO random-walk reproducibility score

Each notebook loads the CNN and the 1/4/8-layer HiCFoundation models, scores every test
window (Input-vs-Target as the baseline plus each model's Predicted-vs-Target), caches
the arrays to `results/score_cache/*.npz` via `load_or_compute_scores`, and produces a
5-way violin plot (with Wilcoxon significance brackets) and a CNN-vs-best-HiCFoundation
scatter under `results/plots/<metric>_analysis/`.

**Before running:** update the `cnn_run_date` / `hicf_run_date` variables near the top of
each notebook to match the dated folders your step-4 runs actually produced. Delete the
matching `results/score_cache/*.npz` if you retrain and want fresh scores.

`results/notebook.ipynb` is the chronological lab notebook (subsample sweeps, loss
curves, prediction montages); it references specific historical run dates and is a record
rather than a clean entry point.

---

## Configuration files

| File | Purpose | Key fields |
|------|---------|-----------|
| `configs/Gse303006.json` | mouse CHARM dataset | `pairs_suffix` `.pairs`, `bin_sizes` [250k, 100k, 50k], `cell_type_dirs` (label → folder) |
| `configs/Gse240114.json` | human LiMCA dataset | `pairs_suffix` `.allValidPairs.txt`, `bin_sizes` [1m, 500k, 250k], `cell_type_dirs` |
| `configs/submatrix_split.json` | window dataset split | `train_val_cell_types`, `test_cell_type`, `val_chroms`, `test_chroms`, `bin_size`, `stride`, `window` |
| `configs/training_hyperparameters.json` | shared training hyperparameters | `batch_size` 64, `lr` 1e-4, `num_epochs` 50, `patience` 3, `num_workers` 0, `hidden_channels` 32 |

`genome_build` in the dataset configs is a label only; no code reads it.

---

## Models

Defined in `src/hic_model.py`; the HiCFoundation pieces live in
`src/hicfoundation_standalone.py`.

**`SimpleEnhanceCNN`** — `Conv2d(1→32) → ReLU → Conv2d(32→32) → ReLU → Conv2d(32→1)`,
all `kernel_size=3, padding=1`. No pooling/upsampling: 256×256 → 256×256, refining values
in place. No output activation (predictions can go slightly negative; downstream code
clips to ≥ 0 before `log1p`).

**`HiCFoundationHead`** wraps `HiCFoundationResEnhancement`:

- **Encoder** (`HiCFoundationModel`, frozen): ViT-L, `embed_dim=1024`, `depth=24`,
  `num_heads=16`, `patch_size=16`. Handles the full upstream input pipeline internally
  (`log10(x+1)` → per-sample max-normalize → invert → fake-RGB `[ones, x, x]` →
  ImageNet-normalize) and a sinusoidal "count" token. Returns patch tokens only.
- **Head** (trainable): `Linear(1024→512)` bridge → `decoder_layers` plain transformer
  blocks (`decoder_dim=512`, `decoder_heads=8`) → `LayerNorm` → `Linear(512 → 16·16)`
  pixel head predicting each patch's block → reassemble to 256×256 → `(out + outᵀ)/2`.
- `decoder_layers=0` is a fully-connected-only head; 1 / 4 / 8 add transformer blocks.
  `freeze_encoder=True` is required (full fine-tuning needs far more VRAM).

---

## Evaluation

All three metrics are reimplemented in `src/hic_train.py` (no external HiCRep/GenomeDISCO
dependency):

- **SSIM** — `skimage.metrics.structural_similarity`, single-scale, on `log1p(clip(·, 0))`
  values, `data_range` from the target. (MS-SSIM was tried and rejected — its downsampling
  washed out real differences.)
- **HiCRep** — box-filter smoothing (`h=1`) then per-diagonal Pearson correlation up to
  `max_bins=200`, combined with variance-stabilized weights `n·(1 + 1/n)/12`.
- **GenomeDISCO** — zero the diagonal, row-normalize to a transition matrix, run random
  walks of length `t=3`, score `1 − ‖Aᵗ − Bᵗ‖₁ / (#non-empty rows)`.

Helpers: `compute_*_scores(model, dataset, device)` for model outputs;
`load_or_compute_scores(cache_path, fn, *args)` for the `score_cache/` disk cache;
`plot_ssim_violin_5way`, `plot_violin_grid`, `plot_score_scatter`, `plot_loss_curve`,
`plot_prediction_comparison`.

---

## Directory and naming conventions

- **Dated folders** — `YYYY-MM-DD` marks either data acquisition date (`data/2026-07-15/`,
  `results/matrices/2026-07-15/`) or a training run date (`results/training/2026-07-30/`).
- **Bin-size labels** — `bin_size_label()`: `1000000 → 1mb`, `500000 → 500kb`,
  `50000 → 50kb`. One subfolder per resolution.
- **Fraction labels** — `frac_label()`: `(0.05, 0.05) → frac_0.05`,
  `(0.1, 0.4) → frac_0.1-0.4`.
- **Cell-type path safety** — `safe_dirname()` replaces `/` with `_` everywhere a cell
  type name becomes a file or directory name (`Ex_L3/4_IT` → `Ex_L3_4_IT`); dict lookups
  keyed by the original label are unaffected.
- **Pooled matrix names** — `<safe celltype>_pbulk.npz`.

---

## Known caveats

- **Config filename case.** The files on disk are `configs/Gse240114.json` and
  `configs/Gse303006.json` (lowercase `se`). Some notebook/script docstrings write
  `GSE...json`. This is harmless on Windows/macOS (case-insensitive filesystems) but
  would break on Linux — use the exact on-disk names, or rename the files and update the
  references.
- **`torch.compile` checkpoints.** Some HiCFoundation runs were saved through
  `torch.compile`, prefixing state-dict keys with `_orig_mod.`. The analysis notebooks
  strip this prefix and load with `strict=False`; keep that handling if you add new load
  sites.
- **Run dates are not auto-discovered.** The analysis notebooks hardcode the training-run
  dates. Update them after retraining.
- **`num_workers=0`** in the default hyperparameters (Windows-friendly). Raise it on
  Linux for faster data loading.
- **Manuscript** (`doc/pbulk-res-enh.tex`) — Methods/Results/Discussion are still stubs.

---

## Authors and citation

Amir Momoh, Ghulam Murtaza, William Stafford Noble — Department of Genome Sciences (and
Paul G. Allen School of Computer Science & Engineering), University of Washington.

Repository: <https://github.com/Noble-Lab/2026_amomoh_pbulk-res-enh>

The standalone HiCFoundation encoder + resolution-enhancement head
(`src/hicfoundation_standalone.py`) was provided by Ghulam Murtaza. HiCFoundation is a
separate project; cite its paper and use its weights per that project's terms.

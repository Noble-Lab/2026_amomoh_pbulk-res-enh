"""
hic_train.py

Training loop pieces for the resolution-enhancement model. Each function
does one job, train_one_epoch updates weights, evaluate checks performance 
without updating weights, train_model runs both across many epochs with early stopping, 
plot_loss_curve visualizes the result.
"""


import copy
import os
import torch
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wilcoxon
from tqdm import tqdm
import time


def load_or_compute_scores(cache_path, compute_fn, *args, **kwargs):
    # Generic cache wrapper for any compute_*_scores function (compute_ssim_scores,
    # compute_hicrep_scores, compute_genomedisco_scores, or any future one with the
    # same "returns array(s) of per-window scores" shape). If cache_path already
    # exists, loads and returns straight from disk -- no model forward passes, no
    # GPU needed, no waiting. Otherwise calls compute_fn(*args, **kwargs), saves the
    # result to cache_path, and returns it. Solves the "restarted the kernel, now
    # everything needs recomputing" problem: once a score set is cached, later
    # notebook runs (even after a full kernel restart) load it in under a second.
    #
    # Works for both single-array returns (compute_hicrep_scores, compute_genomedisco_scores)
    # and 2-array-tuple returns (compute_ssim_scores's (input_scores, pred_scores)) --
    # detected automatically from what compute_fn returns the first time.
    if os.path.exists(cache_path):
        print(f"Loading cached scores from {cache_path}")
        data = np.load(cache_path)
        arrays = [data[key] for key in data.files]
        return arrays[0] if len(arrays) == 1 else tuple(arrays)

    print(f"No cache at {cache_path} -- computing...")
    result = compute_fn(*args, **kwargs)

    save_dir = os.path.dirname(cache_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok = True)

    if isinstance(result, tuple):
        np.savez(cache_path, *result)
    else:
        np.savez(cache_path, result)

    print(f"Saved scores to {cache_path}")
    return result


def _hicrep_mean_filter(matrix: np.ndarray, h: int) -> np.ndarray:
    if h <= 0:
        return matrix
    def _box_filter_1d_sum(x: np.ndarray, h: int, axis: int) -> np.ndarray:
        if h <= 0:
            return x
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (h, h)
        xp = np.pad(x, pad_width, mode='constant')
        n = x.shape[axis]
        out = np.take(xp, range(0, n), axis=axis).copy()
        for j in range(1, 2 * h + 1):
            out += np.take(xp, range(j, j + n), axis=axis)
        return out

    def _valid_neighbor_count(size: int, h: int) -> np.ndarray:
        idx = np.arange(size)
        dist_to_edge = np.minimum(idx, (size - 1) - idx)
        return h + 1 + np.minimum(dist_to_edge, h)

    row_sum = _box_filter_1d_sum(matrix.astype(float, copy=False), h, axis=0)
    total_sum = _box_filter_1d_sum(row_sum, h, axis=1)
    row_count = _valid_neighbor_count(matrix.shape[0], h)
    col_count = _valid_neighbor_count(matrix.shape[1], h)
    return total_sum / (row_count[:, None] * col_count[None, :])


def _hicrep_variance_stabilized_rank_variance(size: int) -> float:
    return np.nan if size < 2 else (1 + 1 / size) / 12


def compute_hicrep(matrix_a: np.ndarray, matrix_b: np.ndarray, h: int = 1, max_bins: int = 200, zero_tol: float = 0.0) -> float:
    smoothed_a = _hicrep_mean_filter(matrix_a, h)
    smoothed_b = _hicrep_mean_filter(matrix_b, h)

    correlations = []
    weights = []

    for diagonal_offset in range(1, min(max_bins, matrix_a.shape[0])):
        diagonal_a = np.diagonal(smoothed_a, offset=diagonal_offset)
        diagonal_b = np.diagonal(smoothed_b, offset=diagonal_offset)
        mask = (
            np.isfinite(diagonal_a)
            & np.isfinite(diagonal_b)
            & ((np.abs(diagonal_a) > zero_tol) | (np.abs(diagonal_b) > zero_tol))
        )
        diagonal_a = diagonal_a[mask]
        diagonal_b = diagonal_b[mask]
        if diagonal_a.size <= 2:
            continue

        correlations.append(np.corrcoef(diagonal_a, diagonal_b)[0, 1])
        weights.append(diagonal_a.size * _hicrep_variance_stabilized_rank_variance(diagonal_a.size))

    correlations = np.nan_to_num(np.array(correlations), copy=True, posinf=0.0, neginf=0.0)
    weights = np.nan_to_num(np.array(weights), copy=True, posinf=0.0, neginf=0.0)
    if weights.sum() == 0:
        return 0.0
    return float(correlations @ weights / weights.sum())


def compute_hicrep_scores(model, dataset, device, batch_size=64, h=1, max_bins=200):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    hicrep_scores = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Computing HiCRep scores"):
            x_dev = x.to(device)
            pred = model(x_dev).cpu()

            for i in range(x.shape[0]):
                pred_mat = np.clip(pred[i, 0].numpy(), 0, None)
                target_mat = np.clip(y[i, 0].numpy(), 0, None)

                score = compute_hicrep(pred_mat, target_mat, h=h, max_bins=max_bins)
                hicrep_scores.append(score)

    return np.array(hicrep_scores)


def _genomedisco_transition(mat: sps.csr_matrix) -> sps.csr_matrix:
    # Row-normalize to a transition matrix; rows summing to zero stay zero.
    sums = np.asarray(mat.sum(axis = 1)).flatten()
    return sps.spdiags(1.0 / np.where(sums == 0, 1.0, sums), [0], mat.shape[0], mat.shape[1], format = "csr").dot(mat)


def compute_genomedisco(matrix_a: np.ndarray, matrix_b: np.ndarray, transition: bool = True, tmax: int = 3, tmin: int = 3) -> float:
    # GenomeDISCO reproducibility score: random walks of length tmin..tmax are run on both
    # matrices and their L1 difference at each step is normalized by the number of non-empty
    # rows. Returns a score in [0, 1] where 1 means identical. Ported from evals.py.
    assert matrix_a.ndim == 2, f"matrix_a has {matrix_a.ndim} dimensions instead of 2"
    assert matrix_b.ndim == 2, f"matrix_b has {matrix_b.ndim} dimensions instead of 2"
    assert matrix_a.shape == matrix_b.shape, "matrices must have the same shape"
    assert 1 <= tmin <= tmax, "expected 1 <= tmin <= tmax"

    # copy before zeroing the diagonals so we don't mutate caller-owned arrays
    mat_a = sps.csr_matrix(matrix_a.copy())
    mat_b = sps.csr_matrix(matrix_b.copy())
    mat_a.setdiag(0)
    mat_b.setdiag(0)
    mat_a.eliminate_zeros()
    mat_b.eliminate_zeros()

    if transition:
        mat_a = _genomedisco_transition(mat_a)
        mat_b = _genomedisco_transition(mat_b)

    nonzero_count = 0.5 * (
        np.count_nonzero(np.asarray(mat_a.sum(axis = 1))) + np.count_nonzero(np.asarray(mat_b.sum(axis = 1)))
    )

    if nonzero_count == 0:
        return 1.0

    scores = []

    for t in range(1, tmax + 1):
        if t == 1:
            walk_a = mat_a.copy()
            walk_b = mat_b.copy()
        else:
            walk_a = walk_a.dot(mat_a)
            walk_b = walk_b.dot(mat_b)

        if t >= tmin:
            scores.append(float(abs(walk_a - walk_b).sum()) / nonzero_count)

    if tmin == tmax:
        return 1 - min(scores[0], 2)

    return 1 - np.trapz(scores, range(len(scores))) / (len(scores) - 1)


def compute_genomedisco_scores(model, dataset, device, batch_size = 64):
    model.eval()
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    gdisco_scores = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc = "Computing GenomeDISCO scores"):
            x_dev = x.to(device)
            pred = model(x_dev).cpu()

            for i in range(x.shape[0]):
                pred_mat = np.clip(pred[i, 0].numpy(), 0, None)
                target_mat = np.clip(y[i, 0].numpy(), 0, None)

                gdisco_scores.append(compute_genomedisco(pred_mat, target_mat))

    return np.array(gdisco_scores)


def set_seed(seed):
    # Seeds model weights initialization.
    torch.manual_seed(seed)             # CPU
    torch.cuda.manual_seed_all(seed)    # GPU


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    # One full pass through the loader, updating model weights using backpropagation
    # on every batch. Returns the average loss across all batches.

    model.train()

    total_loss = 0.0
    n_batches = 0

    timings = {"data_load": 0, "forward": 0, "backward": 0, "step": 0}
    batch_start = time.time()

    for x, y in tqdm(loader, desc="Training", leave=False):
        data_time = time.time() - batch_start

        t0 = time.time()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        t1 = time.time()
        prediction = model(x)           # forward pass
        loss = loss_fn(prediction, y)
        forward_time = time.time() - t1

        t2 = time.time()
        loss.backward()                 # backward pass
        backward_time = time.time() - t2

        t3 = time.time()
        optimizer.step()                # update weights
        step_time = time.time() - t3

        total_loss += loss.item()
        n_batches += 1

        timings["data_load"] += data_time
        timings["forward"] += forward_time
        timings["backward"] += backward_time
        timings["step"] += step_time

        batch_start = time.time()

    avg_loss = total_loss / n_batches
    if n_batches > 0:
        total_time = sum(timings.values())
        tqdm.write(f"\nTiming breakdown ({n_batches} batches):")
        tqdm.write(f"  Data loading: {timings['data_load']:.2f}s ({100*timings['data_load']/total_time:.1f}%)")
        tqdm.write(f"  Forward pass: {timings['forward']:.2f}s ({100*timings['forward']/total_time:.1f}%)")
        tqdm.write(f"  Backward pass: {timings['backward']:.2f}s ({100*timings['backward']/total_time:.1f}%)")
        tqdm.write(f"  Optimizer step: {timings['step']:.2f}s ({100*timings['step']/total_time:.1f}%)")
        tqdm.write(f"  Total: {total_time:.2f}s, {n_batches/total_time:.2f} batches/sec\n")

    return avg_loss


def evaluate(model, loader, loss_fn, device):
    # Same forward pass + loss calculation as train_one_epoch, but no
    # backward() or optimizer.step(), weights are never updated during evaluation.

    # Used for both validation (during training) and testing (at the end)
    # just pointed at different DataLoaders.
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():  # do not track the gradients (nothing here needs backpropagation)
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)

            prediction = model(x)           # forward pass
            loss = loss_fn(prediction, y)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, max_epochs = 50, patience = 3):
    # Calls train_one_epoch + evaluate repeatedly, tracking both their losses.
    # Stops early if val loss has not improved in `patience` epochs
    # this is my safeguard for overfitting: train loss can keep dropping while
    # val loss stalls or rises, and that is the signal to stop.
    # Keeps a copy of the best-val-loss weights in memory and restores them
    # at the end, so the returned model is the best one seen, not just the last.

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_rmse": [], "val_loss": [], "val_rmse": []}

    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_rmse"].append(train_loss ** 0.5)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_loss ** 0.5)


        tqdm.write(f"Epoch {epoch}: Train = {train_loss:.4f} (RMSE = {train_loss ** 0.5:.4f}) "
                   f"Val = {val_loss:.4f} (RMSE = {val_loss ** 0.5:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            tqdm.write(f"No val improvement for {patience} epochs, stopping early at epoch {epoch}")
            break

    model.load_state_dict(best_state)  # restore the best-performing weights, not just the last epoch's

    return model, history


def plot_prediction_comparison(model, dataset, device, n_examples = 5, seed = 42, title = None, save_path = None):
    # n_examples spread across available chromosomes, randomly picked from
    # non-overlapping candidates (windows overlap 255/256 bins at stride = 1).
    # Seeded for reproducibility across model comparisons.

    model.eval()
    rng = np.random.default_rng(seed)

    available = sorted(set(dataset.chroms))
    n_available = len(available)

    base, reminder = divmod(n_examples, n_available)
    counts = [base + (1 if i < reminder else 0) for i in range(n_available)]

    window = dataset.input.shape[1]  # e.g. 256 used to space out non-overlapping candidates

    rows = [] # (chrom, index) pairs, in plotting order
    for chrom, count in zip(available, counts):
        indexes = dataset.indices_for_chrom(chrom)

        if not indexes:
            continue

        # Thin to non-overlapping candidates (one every `window` positions along
        # this chromosome's windows), THEN randomly pick among those candidates
        candidates = indexes[::window] if len(indexes) > window else indexes
        n_pick = min(count, len(candidates))
        chosen = rng.choice(candidates, size = n_pick, replace = False)
        for c in chosen:
            rows.append((chrom, int(c)))

    fig, axes = plt.subplots(len(rows), 3, figsize = (9, 3 * len(rows)), gridspec_kw = {"wspace": 0.15, "hspace": 0.25})

    if len(rows) == 1:
        axes = axes[None, :]  # keep 2D indexing consistent even for a single row
 
    for row, (chrom, index) in enumerate(rows):
        x, y = dataset[index]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)
 
        panels = [("Input", x), ("Predicted", pred), ("Target", y)]

        for col, (label, img) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.log1p(np.clip(img.squeeze().numpy(), 0, None)), cmap="Reds")
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(label, fontweight = "bold")

            if col == 0:
                ax.set_ylabel(chrom, fontweight = "bold", rotation = 0, labelpad = 35)

    if title is not None:
        fig.suptitle(title, fontweight = "bold", y = 1, fontsize = 16)
 
    plt.tight_layout()
    fig.subplots_adjust(wspace = 0.15, hspace = 0.25)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok = True)

        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")

    plt.show()


def compute_ssim_scores(model, dataset, device, batch_size = 64):
    # Runs the model over the WHOLE dataset, and computes SSIM(input, target) as a baseline
    # and SSIM(predicted, target) for the model's actual output, per window. Standard
    # single-scale SSIM (skimage), matching the metric used in the HiCFoundation paper --
    # MS-SSIM was tried but its multiscale downsampling washed out real differences
    # (e.g. reported near-1.0 scores for outputs that were visibly imperfect).

    model.eval()
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    input_ssims = []
    pred_ssims = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Computing SSIM scores"):
            x_dev = x.to(device)
            pred = model(x_dev).cpu()

            for i in range(x.shape[0]):

                # Clip to >= 0 before log1p: the model's raw output is not constrained to be
                # non-negative (last layer has no activation), but contact counts physically
                # can't be negative, and log1p(x) is undefined for x <= -1, an unclipped
                # negative prediction silently produces NaN and corrupts the whole SSIM score.
                in_img = np.log1p(np.clip(x[i, 0].numpy(), 0, None))
                pred_img = np.log1p(np.clip(pred[i, 0].numpy(), 0, None))
                tgt_img = np.log1p(np.clip(y[i, 0].numpy(), 0, None))

                data_range = tgt_img.max() - tgt_img.min()

                if data_range == 0:
                    input_ssims.append(1.0)
                    pred_ssims.append(1.0)
                    continue

                input_ssims.append(ssim(in_img, tgt_img, data_range = data_range))
                pred_ssims.append(ssim(pred_img, tgt_img, data_range = data_range))

    return np.array(input_ssims), np.array(pred_ssims)


def plot_ssim_violin(input_ssims, pred_ssims, title = None, save_path = None):
    # Violin plot comparing SSIM(input, target) vs SSIM(predicted, target)
    # distributions across every window in the dataset and shows whether the
    # model's output is genuinely closer to target than the raw degraded
    # input was, not just on a handful of cherry-picked examples.
 
    fig, ax = plt.subplots(figsize = (10, 6))
    parts = ax.violinplot([input_ssims, pred_ssims], showmeans = True, showmedians = True)

    colors = ["royalblue", "#FF4D53"]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
 
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Input vs Target\n(mean = {input_ssims.mean():.3f})",
                         f"Predicted vs Target\n(mean = {pred_ssims.mean():.3f})"])
    ax.set_ylabel("SSIM", fontweight = "bold")
    ax.set_title(title or "SSIM Distribution: Input vs Predicted (against Target)")
    ax.grid(True, alpha = 0.3, axis = "y")
    plt.tight_layout()
 
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok = True)

        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")
 
    plt.show()


def plot_ssim_violin_3way(input_ssims, cnn_pred_ssims, hicfoundation_pred_ssims, title = None, save_path = None):
    # Three-way violin: SSIM(input, target) vs SSIM(CNN predicted, target)
    # vs SSIM(HiCFoundation-head predicted, target), same test set, same
    # target -- so all three distributions are directly comparable window
    # for window. input_ssims only needs to be computed once (same input
    # data for both models); pass whichever one you already have.
    fig, ax = plt.subplots(figsize = (12, 6))
    parts = ax.violinplot(
        [input_ssims, cnn_pred_ssims, hicfoundation_pred_ssims], showmeans = True, showmedians = True
        )

    labels = ["Input", "CNN", "HiCFoundation"]
    colors = ["#7F7F7F", "royalblue", "#FF4D53"]

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([
        f"Input vs Target\n(mean = {input_ssims.mean():.3f}, median = {np.median(input_ssims):.3f})",
        f"CNN Prediction vs Target\n(mean = {cnn_pred_ssims.mean():.3f}, median = {np.median(cnn_pred_ssims):.3f})",
        f"HiCFoundation Prediction vs Target\n(mean = {hicfoundation_pred_ssims.mean():.3f}, median = {np.median(hicfoundation_pred_ssims):.3f})"
        ])

    ax.set_ylabel("SSIM", fontweight = "bold")
    ax.set_title(title or "SSIM Distribution: Input vs CNN vs HiCFoundation (against Target)", fontweight = "bold")
    ax.grid(True, alpha = 0.3, axis = "y")
 
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor = color, alpha = 0.6) for color in colors]
    ax.legend(legend_handles, labels, loc = "best")
 
    plt.tight_layout()
 
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok = True)
 
        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")
 
    plt.show()


def plot_ssim_violin_5way(scores, save_path = None, pairs = None, alpha = 0.05, metric_name = "Score"):
    # scores: dict of label -> array of per-window scores. No title (poster
    # panels are captioned externally), no whiskers, no median line -- only
    # the mean is shown, written directly on each violin. Labels sit
    # horizontal (no rotation). Y-axis fixed to [0, 1].
    labels = list(scores.keys())
    n = len(labels)

    if pairs is None:
        pairs = list(zip(labels[:-1], labels[1:]))

    # Format labels for display -- singular "Layer" for 1, plural "Layers" otherwise
    display_labels = []
    for label in labels:
        if "HiCFoundation" in label and "-layer" in label:
            layers = label.split("(")[1].split("-")[0]
            unit = "Layer" if layers.strip() == "1" else "Layers"
            display_labels.append(f"HiCFoundation\n({layers}-{unit})")
        else:
            display_labels.append(label)

    palette = {
        "Input": "#7F7F7F",
        "CNN": "royalblue",
    }
    coral_shades = ["#FF9498", "#FF4D53", "#BF3A3E"]
    coral_i = 0
    colors = []

    for label in labels:
        if label in palette:
            colors.append(palette[label])
        else:
            colors.append(coral_shades[min(coral_i, len(coral_shades) - 1)])
            coral_i += 1

    fig, ax = plt.subplots(figsize = (13, 5.5))

    positions = range(1, n + 1)
    parts = ax.violinplot([scores[label] for label in labels], positions = positions,
                           showmeans = False, showmedians = False, showextrema = False)

    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")

    # Mean written directly on each violin, underlined (no background box) --
    # no whiskers, no median line.
    for pos, label in zip(positions, labels):
        mean_val = float(np.mean(scores[label]))
        ax.text(pos, mean_val, f"{mean_val:.3f}", ha = "center", va = "center",
                 fontsize = 11, fontweight = "bold", color = "black", zorder = 3)
        ax.plot([pos - 0.13, pos + 0.13], [mean_val - 0.028, mean_val - 0.028],
                 color = "royalblue", linewidth = 1.5, zorder = 3)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(display_labels, fontsize = 16, fontweight = "bold", rotation = 0, ha = "center")
    ax.set_ylabel(metric_name, fontweight = "bold", fontsize = 16)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis = "y", labelsize = 13)
    ax.grid(True, axis = "y", alpha = 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(bottom = 0.15, top = 0.8)

    # Brackets are drawn in plain data coordinates now (not axes-fraction) so
    # the left spine -- which matplotlib draws across the full y-limit range --
    # naturally extends up to cover them instead of floating disconnected
    # above a spine that stops at 1.0. No extra tick labels appear above 1.0;
    # the axis line just runs a bit further to visually reach the brackets.
    bracket_height = 1.03

    for label_a, label_b in pairs:
        i = labels.index(label_a) + 1
        j = labels.index(label_b) + 1

        stat, p_value = wilcoxon(scores[label_a], scores[label_b])
        stars = "*" if p_value < alpha else "-"

        ax.plot([i, i, j, j],
                 [bracket_height, bracket_height + 0.03,
                  bracket_height + 0.03, bracket_height],
                 color = "black", linewidth = 1, clip_on = False)
        ax.text((i + j) / 2, bracket_height + 0.035, stars, ha = "center", va = "bottom",
                 fontweight = "bold")

        bracket_height += 0.07

    ax.set_ylim(0, bracket_height + 0.02)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok = True)
        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")

    plt.show()


def plot_violin_grid(metrics, save_path = None, pairs = None, alpha = 0.05):
    # metrics: dict of metric_name -> {label: array of per-window scores},
    # e.g. {"SSIM": {...}, "HiCRep": {...}, "GenomeDISCO": {...}}. Same label
    # set/order expected in every metric's dict, since they share one x-axis.
    # Stacks one subplot per metric, top to bottom in dict order; only the
    # bottom subplot gets x-tick labels. Same styling as plot_ssim_violin_5way:
    # no title, no grid, no whiskers, no median line, no top/right spines --
    # mean written directly on each violin with an underline, each subplot's
    # y-axis extends past 1.0 to cover its own significance brackets.
    metric_names = list(metrics.keys())
    labels = list(next(iter(metrics.values())).keys())
    n = len(labels)

    if pairs is None:
        pairs = list(zip(labels[:-1], labels[1:]))

    display_labels = []
    for label in labels:
        if "HiCFoundation" in label and "-layer" in label:
            layers = label.split("(")[1].split("-")[0]
            unit = "Layer" if layers.strip() == "1" else "Layers"
            display_labels.append(f"Hi-C Foundation\n({layers}-{unit})")
        else:
            display_labels.append(label)

    palette = {"Input": "#7F7F7F", "CNN": "royalblue"}
    coral_shades = ["#FF9498", "#FF4D53", "#BF3A3E"]

    coral_i = 0
    colors = []

    for label in labels:
        if label in palette:
            colors.append(palette[label])
        else:
            colors.append(coral_shades[min(coral_i, len(coral_shades) - 1)])
            coral_i += 1

    fig, axes = plt.subplots(len(metric_names), 1, figsize = (13, 5.5 * len(metric_names)), sharex = True)

    if len(metric_names) == 1:
        axes = [axes]

    positions = range(1, n + 1)

    for ax, metric_name in zip(axes, metric_names):
        scores = metrics[metric_name]

        parts = ax.violinplot([scores[label] for label in labels], positions = positions,
                               showmeans = False, showmedians = False, showextrema = False)

        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
            body.set_edgecolor("black")

        # Mean written directly on each violin, underlined (no background box) --
        # no whiskers, no median line -- matching plot_ssim_violin_5way.
        for pos, label in zip(positions, labels):
            mean_val = float(np.mean(scores[label]))
            ax.text(pos, mean_val, f"{mean_val:.3f}", ha = "center", va = "center",
                     fontsize = 11, fontweight = "bold", color = "black", zorder = 3)
            ax.plot([pos - 0.13, pos + 0.13], [mean_val - 0.028, mean_val - 0.028],
                     color = "black", linewidth = 1.5, zorder = 3)

        ax.set_ylabel(metric_name, fontweight = "bold", fontsize = 16)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis = "y", labelsize = 13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Brackets in plain data coordinates (not axes-fraction), same as
        # plot_ssim_violin_5way -- the left spine naturally extends to cover
        # them since it spans the full (dynamically set) y-limit range.
        bracket_height = 1.03

        for label_a, label_b in pairs:
            i = labels.index(label_a) + 1
            j = labels.index(label_b) + 1

            stat, p_value = wilcoxon(scores[label_a], scores[label_b])
            mark = "*" if p_value < alpha else "-"

            ax.plot([i, i, j, j],
                     [bracket_height, bracket_height + 0.03,
                      bracket_height + 0.03, bracket_height],
                     color = "black", linewidth = 1, clip_on = False)
            ax.text((i + j) / 2, bracket_height + 0.035, mark,
                     ha = "center", va = "bottom", fontweight = "bold")

            bracket_height += 0.07

        ax.set_ylim(0, bracket_height + 0.02)

    axes[-1].set_xticks(list(positions))
    axes[-1].set_xticklabels(display_labels, fontsize = 16, fontweight = "bold", rotation = 0, ha = "center")

    for ax in axes[:-1]:
        ax.tick_params(labelbottom = False)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok = True)

        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")

    plt.show()


def plot_score_scatter(scores_x, scores_y, label_x, label_y, metric_name = "SSIM", save_path = None):
    # Paired per-window scatter: scores_x and scores_y must be the same length
    # and index-aligned (same windows, same order), same requirement as
    # plot_ssim_violin_5way's wilcoxon pairing. Plain single-color scatter
    # with a y = x reference line -- no fit line, no Pearson r annotation,
    # no legend, no title. Works for SSIM, HiCRep, or GenomeDISCO scores (or
    # any other paired per-window metric); metric_name only changes the
    # axis labels. Axes are fixed to the full [0, 1] range (the theoretical
    # bounds for these metrics), same scale on both axes regardless of where
    # the real data falls, for direct comparability across different metric
    # scatter plots. Top-left box reports how many windows label_y (e.g. the
    # HiCFoundation variant) scored higher than label_x (e.g. CNN).
    fig, ax = plt.subplots(figsize = (10, 6))

    ax.plot([0, 1], [0, 1], color = "black", linewidth = 1, linestyle = "--", alpha = 0.6, zorder = 1)
    ax.scatter(scores_x, scores_y, s = 10, alpha = 0.4, color = "#FF4D53", zorder = 2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_xlabel(f"{label_x} {metric_name}", fontweight = "bold", fontsize = 12)
    ax.set_ylabel(f"{label_y} {metric_name}", fontweight = "bold", fontsize = 12)
    ax.set_aspect("equal")
    ax.grid(True, alpha = 0.3)

    # Win-count box: how many windows label_y (e.g. HiCFoundation) scored
    # higher than label_x (e.g. CNN) on this metric, out of the total.
    wins = int((scores_y > scores_x).sum())
    total = len(scores_y)
    pct = 100 * wins / total if total else 0.0
    ax.text(0.03, 0.97,
            f"{label_y} outperforms {label_x} in {wins:,}/{total:,} windows ({pct:.1f}%)",
            transform = ax.transAxes, ha = "left", va = "top",
            fontsize = 10, fontweight = "bold",
            bbox = dict(boxstyle = "round", facecolor = "white", edgecolor = "black", alpha = 0.9))

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok = True)
        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")

    plt.show()

def plot_loss_curve(history, title = None, save_path = None):
    # history: the dict returned by train_model (keys "train_loss", "val_loss").
    # Where the two curves diverge is where overfitting starts.
    # Where val loss flattens is roughly where the model has converged.

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(epochs, history["train_loss"], color = "royalblue", linewidth = 2, label = "Train loss")
    ax.plot(epochs, history["val_loss"], color = "#FF4D53", linewidth = 2, linestyle = "--", label = "Validation loss")
    ax.set_title(title or "Training vs Validation Loss", fontweight = "bold") 
    ax.set_xlabel("Epoch", fontweight = "bold")
    ax.set_ylabel("Loss (MSE)", fontweight = "bold")
    ax.legend()
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok = True)

        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")

    plt.show()
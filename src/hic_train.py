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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim


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

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()           # clear gradients from the previous batch

        prediction = model(x)           # forward pass
        loss = loss_fn(prediction, y)

        loss.backward()                 # backward pass (computes the gradients)

        optimizer.step()                # update weights using those gradients

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, loss_fn, device):
    # Same forward pass + loss calculation as train_one_epoch, but no
    # backward() or optimizer.step(), weights are never updated during evaluation.

    # Used for both validation (during training) and testing (at the end)
    # just pointed at different DataLoaders.    
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():  # do not track the gradients (nothing here needs backpropagation)
        for x, y in loader:
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

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_rmse"].append(train_loss ** 0.5)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_loss ** 0.5)


        print(f"Epoch {epoch}: Train = {train_loss:.4f} (RMSE = {train_loss ** 0.5:.4f}) "
              f"Val = {val_loss:.4f} (RMSE = {val_loss ** 0.5:.4f})")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"No val improvement for {patience} epochs, stopping early at epoch {epoch}")
            break

    model.load_state_dict(best_state)  # restore the best-performing weights, not just the last epoch's

    return model, history


def plot_prediction_comparison(model, dataset, device, n_examples = 5, seed = 42, save_path = None):
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

    fig, axes = plt.subplots(len(rows), 3, figsize = (9, 3 * len(rows)))

    if len(rows) == 1:
        axes = axes[None, :]  # keep 2D indexing consistent even for a single row
 
    for row, (chrom, index) in enumerate(rows):
        x, y = dataset[index]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)
 
        panels = [("Input", x), ("Predicted", pred), ("Target", y)]

        for col, (label, img) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.log1p(img.squeeze().numpy()), cmap="Reds")
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(label, fontweight = "bold")

            if col == 0:
                ax.set_ylabel(chrom, fontweight = "bold", rotation = 0, labelpad = 35)
 
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok = True)

        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")

    plt.show()


def compute_ssim_scores(model, dataset, device, batch_size = 64):
    # Runs the model over the WHOLE dataset, and computes the SSIM(input, target) as a baseline 
    # and SSIM(predicted, target) for the model's actual output, per window. 
    # Returns two arrays, same length, for plot_ssim_violin() or any other downstream analysis. 

    model.eval()
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    input_ssims = []
    pred_ssims = []
 
    with torch.no_grad():
        for x, y in loader:
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


def plot_loss_curve(history, title = None, save_path = None):
    # history: the dict returned by train_model (keys "train_loss", "val_loss").
    # Where the two curves diverge is where overfitting starts.
    # Where val loss flattens is roughly where the model has converged.

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(epochs, history["train_loss"], color = "royalblue", linewidth = 2, label = "Train loss")
    ax.plot(epochs, history["val_loss"], color = "#FF4D53", linewidth = 2, linestyle = "--", label = "Validation loss")
    ax.set_title(title or "Training vs Validation Loss")
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
"""
train_model.py

Trains the pseudo-bulk Hi-C resolution enhancement CNN
(SimpleEnhanceCNN, src/hic_model.py) on paired input/target submatrices
built by build_submatrix_dataset.py.

Paths are passed as CLI args and the hyperparameters live in a JSON config.
 
Run: python scripts/train_model.py --config configs/training_hyperparameters.json \
    --train-h5 results/submatrices/train.h5 \
    --val-h5 results/submatrices/val.h5 \
    --out-dir results/training

Loss curve plotting happens separately, in results/notebook.ipynb, by loading
history.json and calling plot_loss_curve(history, save_path = ...) from
hic_train.py directly. Save_path writes the figure to results/plots/<date>/
in addition to displaying it inline.
"""


import argparse
import os
import json
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src/ to the path so this script runs directly without installing a package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from hic_dataset import SubmatrixDataset
from hic_model import SimpleEnhanceCNN
from hic_train import set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser(description = "Train the Hi-C resolution enhancement CNN")
    parser.add_argument("--config", required = True,
                        help = "Hyperparameter config, e.g. configs/training_hyperparameters.json")
    parser.add_argument("--train-h5", required = True, help = "Path to train.h5")
    parser.add_argument("--val-h5", required = True, help = "Path to val.h5")
    parser.add_argument("--out-dir", default = "results/training",
                        help = "Where to write model weights + training history "
                        "(a dated subfolder is created under this)")
    parser.add_argument("--seed", type = int, default = 42,
                        help = "Seed for model weight initialization (set_seed in hic_train.py)")

    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)


def main():
    args = parse_args()
    config = load_config(args.config)

    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 1e-4)
    num_epochs = config.get("num_epochs", 50)
    patience = config.get("patience", 3)
    num_workers = config.get("num_workers", 0)
    hidden_channels = config.get("hidden_channels", 32)

    run_date = time.strftime("%Y-%m-%d")
    out_dir = os.path.join(args.out_dir, run_date)
    os.makedirs(out_dir, exist_ok = True)
 
    model_path = os.path.join(out_dir, "model.pt")
    history_path = os.path.join(out_dir, "history.json")
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SubmatrixDataset(args.train_h5)
    val_dataset = SubmatrixDataset(args.val_h5)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    print(f"Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,}")

    set_seed(args.seed)
    model = SimpleEnhanceCNN(hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.MSELoss()

    # train_model() already keeps the best-val-loss weights in memory (deepcopy)
    # and restores them before returning, so `model` here is the best model,
    # there is no separate best-vs-final checkpoint to track.

    model, history = train_model(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        optimizer = optimizer,
        loss_fn = loss_fn,
        device = device,
        max_epochs = num_epochs,
        patience = patience
    )

    # Plotting happens separately in results/notebook.ipynb:
    #   history = json.load(open("results/training/<date>/history.json"))
    #   plot_loss_curve(history, save_path = "results/plots/<date>/loss_curve.png")
    with open(history_path, "w") as file:
        json.dump(history, file, indent = 2)

    torch.save(model.state_dict(), model_path)

    print(f"Best-val model weights: {model_path}")
    print(f"Training history: {history_path}")

    
if __name__ == "__main__":
    main()
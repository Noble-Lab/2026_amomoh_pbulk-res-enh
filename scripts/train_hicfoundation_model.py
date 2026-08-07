"""
train_hicfoundation_model.py

Trains HiCFoundationHead (src/hic_model.py), a frozen HiCFoundation encoder +
trainable bridge/decoder/pixel_head, on the same paired submatrices used
for SimpleEnhanceCNN, so results are directly comparable.

Paths are passed as CLI args and the hyperparameters live in a JSON config
(the same training_hyperparameters.json used for the CNN and same lr).

Run: python scripts/train_hicfoundation_model.py --config configs/training_hyperparameters.json \
    --train-h5 results/submatrices/train.h5 \
    --val-h5 results/submatrices/val.h5 \
    --hicfoundation-weights hicfoundation_model/hicfoundation_pretrain.pth.tar \
    --out-dir results/training_hicfoundation
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
from hic_model import HiCFoundationHead
from hic_train import set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser(description = "Train the HiCFoundation model")
    parser.add_argument("--config", required = True,
                        help = "Hyperparameter config, e.g. configs/training_hyperparameters.json ")
    parser.add_argument("--train-h5", required = True, help = "Path to train.h5")
    parser.add_argument("--val-h5", required = True, help = "Path to val.h5")
    parser.add_argument("--hicfoundation-weights", required = True,
                        help = "Path to hicfoundation_pretrain.pth.tar")
    parser.add_argument("--decoder-layers", type = int, default = 0,
                        help = "Transformer decoder layers on top of bridge/pixel_head. "
                        "0 (default) = fully-connected-only head,"
                        "Set > 0 for the transformer-decoder variant.")
    parser.add_argument("--out-dir", default = "results/training_hicfoundation",
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
    model = HiCFoundationHead(args.hicfoundation_weights, decoder_layers = args.decoder_layers).to(device)
    # Encoder is frozen inside HiCFoundationResEnhancement (requires_grad = False on all
    # encoder params), so only hand the optimizer the params that actually need updating:
    # bridge, decoder_blocks (if decoder_layers > 0), decoder_norm, pixel_head.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr = lr)
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
    # history = json.load(open("results/training_hicfoundation/<date>/history.json"))
    # plot_loss_curve(history, save_path = "results/plots/<date>/hicfoundation_loss_curve.png")
    with open(history_path, "w") as file:
        json.dump(history, file, indent = 2)

    torch.save(model.state_dict(), model_path)

    print(f"Best-val model weights: {model_path}")
    print(f"Training history: {history_path}")

    
if __name__ == "__main__":
    main()
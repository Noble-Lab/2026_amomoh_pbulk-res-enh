"""
hic_train.py

Training loop pieces for the resolution-enhancement model. Each function
does one job, train_one_epoch updates weights, evaluate checks performance 
without updating weights, train_model runs both across many epochs with early stopping, 
plot_loss_curve visualizes the result.
"""


import copy
import torch
import matplotlib.pyplot as plt


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
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}: train = {train_loss:.4f} val = {val_loss:.4f}")

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


def plot_loss_curve(history, title = None):
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
    plt.show()
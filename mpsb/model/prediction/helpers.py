from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from mpsb.util.consts_and_types import PredictionModelSets

MAX_LOSS_THRESHOLD = 1e5

def gen_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_torch, y_torch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path for saving the best model checkpoint.
                        Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}: Best loss: {self.best_loss:.4f}, current loss: {val_loss:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        self.best_model_state = model.state_dict().copy()
        if self.verbose:
            print(f"Validation loss improved. Saving model...")


def train_and_evaluate_model(
        model: torch.nn.Module,
        data: PredictionModelSets,
        batch_size: int,
        device: torch.device,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        early_stopping: bool = False,
        patience: int = 10,
        delta: float = 0.0,
        checkpoint_path: str = 'checkpoint.pt',
        noise_std: float = 0.0,
        scale_std: float = 1.0,
        verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    train_loader = gen_data_loader(data.X_train, data.y_train, batch_size)
    val_loader = gen_data_loader(data.X_val, data.y_val, batch_size) if data.X_val is not None else None
    augmentation_columns = data.aug_cols

    train_losses = []
    val_losses = []
    early_stopper = None

    if early_stopping:
        early_stopper = EarlyStopping(patience=patience, delta=delta, verbose=verbose, path=checkpoint_path)

    for epoch in range(epochs):
        epoch_loss = _train_model(
            train_loader, model, criterion, optimizer, augmentation_columns, noise_std, scale_std, device)
        train_losses.append(epoch_loss)
        log_str = f"Epoch {epoch + 1}/{epochs} Train Loss: {epoch_loss:.4f}"

        if val_loader is not None:
            val_epoch_loss = _validate_model(val_loader, model, criterion, device)
            val_losses.append(val_epoch_loss)
            log_str += f", Val Loss: {val_epoch_loss:.4f}"

            # Check for early stopping if enabled
            if early_stopping:
                early_stopper(val_epoch_loss, model)
                if early_stopper.early_stop:
                    if verbose: print("Early stopping triggered!")
                    # Load the best model weights
                    model.load_state_dict(early_stopper.best_model_state)
                    break

        if verbose: print(log_str)
    return np.array(train_losses), np.array(val_losses)


def _validate_model(
        val_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device
) -> float:
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)

            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_running_loss += val_loss.item() * val_inputs.size(0)
    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    return val_epoch_loss


def _train_model(
        train_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        aug_cols: List[int],
        noise_std: float,
        scale_std: float,
        device: torch.device
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Prepare model inputs
        inputs = inputs.to(device)  # (batch, seq_len, features)
        targets = targets.to(device)  # (batch, horizon)

        if aug_cols:
            if len(inputs.shape) != 3:
                raise ValueError("Augmentation columns are only supported for 3D inputs.")
            # Add noise to the augmentation columns
            noise = torch.randn_like(inputs[..., aug_cols]) * noise_std
            inputs[..., aug_cols] += noise
            # Scale the augmentation columns
            batch_size = inputs.size(0)
            scale_factor = 1.0 + torch.randn(batch_size, 1, 1, device=device) * scale_std
            inputs[..., aug_cols] *= scale_factor

        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, horizon)
        loss = criterion(outputs, targets)

        # If the loss is extremely large, something is diverging
        if loss.item() > MAX_LOSS_THRESHOLD:
            raise RuntimeError(f"Loss exploded to {loss.item()} > {MAX_LOSS_THRESHOLD}.")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

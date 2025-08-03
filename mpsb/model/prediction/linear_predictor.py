from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from optuna import Trial

from .base_predictor import BasePredictor
from .helpers import train_and_evaluate_model
from ...util.consts_and_types import PredictionModelSets


class LinearMLP(nn.Module):
    """
    A multi-layer perceptron with purely linear layers (no activations).
    Essentially does: out = Wn(...W2(W1 x + b1)... + b2) + bn
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super(LinearMLP, self).__init__()
        layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        in_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd, bias=True))
            # No activation or dropout, just linear
            in_dim = hd
        # Final linear layer from the last hidden dimension to the output
        layers.append(nn.Linear(in_dim, output_dim, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x shape: (batch, input_dim)
        Returns: (batch, output_dim)
        """
        return self.model(x)


class LinearPredictor(BasePredictor):
    """
    A predictor that uses a purely linear MLP (no non-linear activations).
    This is effectively a sequence of linear layers stacked back-to-back.
    """

    def __init__(
            self,
            hidden_dims: list[int] = None,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 20,
            early_stopping: bool = False,
            patience: int = 10,
            delta: float = 0.0,
            checkpoint_path: str = 'checkpoint.pt',
            verbose: bool = True,
            device: str = 'cpu',
            **kwargs
    ):
        """
        Args:
            hidden_dims (list[int]): Sizes of the hidden layers. If empty or None, no hidden layers -> single linear layer.
            learning_rate (float): Learning rate for Adam.
            batch_size (int): Batch size.
            epochs (int): Number of training epochs.
            kwargs: Unused extras for compatibility.
        """
        super().__init__(**kwargs)
        if hidden_dims is None:
            hidden_dims = []

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self._device = torch.device(device)
        self.model: nn.Module = None

    def _build_model(self, input_dim: int, output_dim: int) -> LinearMLP:
        model = LinearMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
        )
        return model.to(self._device)

    def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the linear MLP on the data.
        This expects data.X_train, data.y_train to be 2D arrays of shape:
            - X_train: (N, input_dim)
            - y_train: (N, output_dim)
        If your data is 3D (like (N, seq_len, features)),
        you can flatten it externally or adapt below as needed.
        """
        input_dim = data.X_train.shape[1]
        output_dim = data.y_train.shape[1]

        if self.model is None:
            self.model = self._build_model(input_dim, output_dim)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_losses, val_losses = train_and_evaluate_model(
            model=self.model,
            data=data,
            batch_size=self.batch_size,
            device=self._device,
            epochs=self.epochs,
            optimizer=optimizer,
            criterion=criterion,
            early_stopping=self.early_stopping,
            patience=self.patience,
            delta=self.delta,
            checkpoint_path=self.checkpoint_path,
            verbose=self.verbose,
        )
        return train_losses, val_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for input X of shape (N, input_dim).
        Return shape (N, output_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        self.model.eval()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            preds = self.model(X_torch)
        return preds.cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save model weights and hyperparameters for reproducibility.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.model.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.model.output_dim,
            'learning_rate': self.learning_rate,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load model weights and hyperparameters from file.
        """
        checkpoint = torch.load(path, map_location=self._device)
        input_dim = checkpoint['seq_len']
        self.hidden_dims = checkpoint['hidden_dims']
        output_dim = checkpoint['output_dim']
        self.learning_rate = checkpoint['learning_rate']

        self.model = self._build_model(input_dim, output_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    @staticmethod
    def sample(trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna or other tuning.
        This is purely optional—adjust as needed.
        """
        # Example minimal space
        hidden_dim = trial.suggest_categorical("hidden_dim", [None, 64, 128, 256, 512, 1024])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

        return {
            "hidden_dims": [hidden_dim] if hidden_dim is not None else [],
            "learning_rate": learning_rate,
        }

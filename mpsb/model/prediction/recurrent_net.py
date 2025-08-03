from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from optuna import Trial

from .base_predictor import BasePredictor
from .helpers import train_and_evaluate_model
from ...util.consts_and_types import PredictionModelSets


class RNNModel(nn.Module):
    """
    A simple recurrent model (LSTM or GRU) for multi-step forecasting.
    Predicts 'horizon' steps ahead from the final hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 horizon: int, rnn_type: str = 'LSTM', dropout: float = 0.0):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # Choose the RNN variant
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'.")

        # Final fully connected layer to map hidden state -> horizon outputs
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x shape: (batch, seq_len, input_size)
        Returns:
          (batch, horizon)
        """
        out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_size)
        last_out = out[:, -1, :]  # (batch, hidden_size)
        preds = self.fc(last_out)  # (batch, horizon)
        return preds


class RecurrentPredictor(BasePredictor):
    """
    A recurrent neural network (LSTM or GRU) predictor for multi-step time series forecasting.
    Predicts 'horizon' future steps at once (direct multi-step).
    """

    def __init__(
            self,
            rnn_type: str = 'LSTM',
            units: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 20,
            horizon: int = 96,
            early_stopping: bool = False,
            patience: int = 10,
            delta: float = 0.0,
            checkpoint_path: str = 'checkpoint.pt',
            verbose: bool = True,
            device: str = 'cpu',
            **kwargs
    ):
        super().__init__(**kwargs)  # Ensures BasePredictor init is called

        # Hyperparameters
        self.rnn_type = rnn_type
        self.units = units
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.horizon = horizon
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path

        # Additional user config
        self.verbose = verbose

        # Model internals
        self.model: nn.Module = None
        self.input_size = None  # determined at fit time

        # Device management
        self._device = torch.device(device)

    def _build_model(self, input_size: int) -> nn.Module:
        """Helper to build a new RNNModel instance on the correct device."""
        model = RNNModel(
            input_size=input_size,
            hidden_size=self.units,
            num_layers=self.num_layers,
            horizon=self.horizon,
            rnn_type=self.rnn_type,
            dropout=self.dropout
        )
        return model.to(self._device)

    def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the RNN model for 'epochs' epochs.
        
        Args:
          data: Contains train/validation sets with shapes:
            X_train: (N_train, seq_len, features)
            y_train: (N_train, horizon)
            X_val: (N_val, seq_len, features)  [optional]
            y_val: (N_val, horizon)            [optional]
        
        Returns:
          (train_losses, val_losses) as two np.ndarray of length 'epochs'
            - If no validation data is provided, val_losses will be empty array.
        """
        # Determine input_size from X_train
        # Expecting shape: (batch, seq_len, features)
        self.input_size = data.X_train.shape[2]

        # Build model if not already built
        if self.model is None:
            self.model = self._build_model(self.input_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses, val_losses = train_and_evaluate_model(
            model=self.model,
            data=data,
            batch_size=self.batch_size,
            device=self._device,
            epochs=self.epochs,
            optimizer=optimizer,
            criterion=criterion,
            verbose=self.verbose,
            early_stopping=self.early_stopping,
            patience=self.patience,
            delta=self.delta,
            checkpoint_path=self.checkpoint_path,
        )
        return np.array(train_losses), np.array(val_losses)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the next 'horizon' points for each sequence in X.
        
        Args:
          X: shape (N, seq_len, features)
        
        Returns:
          Numpy array of shape (N, horizon)
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        self.model.eval()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            outputs = self.model(X_torch)  # (N, horizon)

        return outputs.cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save model weights and hyperparameters for reproducibility.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'rnn_type': self.rnn_type,
            'units': self.units,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'horizon': self.horizon,
            'input_size': self.input_size,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load model weights and hyperparameters from file.
        """
        checkpoint = torch.load(path, map_location=self._device)

        # Restore hyperparameters
        self.rnn_type = checkpoint['rnn_type']
        self.units = checkpoint['units']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.learning_rate = checkpoint['learning_rate']
        self.horizon = checkpoint['horizon']
        self.input_size = checkpoint['input_size']

        # Rebuild and load state
        self.model = self._build_model(self.input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    @staticmethod
    def sample(trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna tuning, including which GPU
        and verbosity setting to use. You can adjust these as needed.
        """
        # Typical RNN hyperparameters
        rnn_type = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU'])
        units = trial.suggest_int('units', 32, 128, step=32)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        return {
            'rnn_type': rnn_type,
            'units': units,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
        }

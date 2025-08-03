from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from optuna import Trial

from .base_predictor import BasePredictor
from .helpers import train_and_evaluate_model
from .net.times_net import TimesNetModel  # Your TimesNet "Model" class here
from ...util.consts_and_types import PredictionModelSets


class TimesNetPredictor(BasePredictor):
    """
    Predictor wrapping the TimesNet model.
    Forecasts 'pred_len' steps given 'seq_len' input steps.
    """

    def __init__(
            self,
            seq_len: int = 96,
            pred_len: int = 96,
            top_k: int = 2,
            num_kernels: int = 6,
            e_layers: int = 2,
            d_model: int = 32,
            d_ff: int = 64,
            c_out: int = 1,  # number of output channels/features
            enc_in: int = 1,  # number of input features/channels
            freq: str = 'h',  # frequency for embedding
            embed: str = 'fixed',  # embedding type
            dropout: float = 0.1,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 20,
            early_stopping: bool = False,
            patience: int = 10,
            delta: float = 0.0,
            checkpoint_path: str = 'checkpoint.pt',
            device: str = 'cpu',
            verbose: bool = True,
            **kwargs
    ):
        """
        Args:
            seq_len (int): input sequence length.
            pred_len (int): number of time steps to predict (forecast horizon).
            top_k (int): how many frequencies to pick in FFT_for_Period.
            num_kernels (int): # of Inception kernels per TimesBlock.
            e_layers (int): how many TimesBlocks to stack.
            d_model (int): dimension of the per-time-step embedding.
            d_ff (int): hidden dim in the TimesBlock's convolution layers.
            c_out (int): # of output features (usually 1 if you're predicting just “load”).
            enc_in (int): # of input features per time step.
            freq (str): frequency code for positional embeddings (e.g. 'h' for hour).
            embed (str): embedding type ('fixed' or 'timeF').
            dropout (float): dropout probability.
            learning_rate (float): optimizer learning rate.
            batch_size (int): training mini-batch size.
            epochs (int): training epochs.
            device (str): 'cpu' or 'cuda:0', etc.
            verbose (bool): whether to print train/val loss each epoch.
            kwargs: unused extra arguments to conform with BasePredictor signature.
        """
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.c_out = c_out
        self.enc_in = enc_in
        self.freq = freq
        self.embed = embed
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model: nn.Module = None

    def _build_model(self, target_indexes: np.ndarray) -> nn.Module:
        """
        Build a TimesNet 'Model' on the correct device.
        """
        model = TimesNetModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            c_out=self.c_out,
            enc_in=self.enc_in,
            freq=self.freq,
            embed=self.embed,
            dropout=self.dropout,
            target_indexes=target_indexes,
        )
        return model.to(self._device)

    def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train TimesNet for 'epochs' epochs on the given data.

        data.X_train: (N_train, seq_len, enc_in)
        data.y_train: (N_train, pred_len, c_out) or (N_train, pred_len)
        data.X_val:   (N_val, seq_len, enc_in) [optional]
        data.y_val:   (N_val, pred_len, c_out) [optional]

        Returns: (train_losses, val_losses) as np.ndarray.
        """
        if self.model is None:
            self.model = self._build_model(data.target_indexes)

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
            verbose=self.verbose,
            early_stopping=self.early_stopping,
            patience=self.patience,
            delta=self.delta,
            checkpoint_path=self.checkpoint_path,
        )
        return train_losses, val_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next 'pred_len' steps.
        X: shape (N, seq_len, enc_in)

        Returns:
          np.ndarray of shape (N, pred_len, c_out).
        """
        if self.model is None:
            raise RuntimeError("TimesNetPredictor is not fitted yet.")

        self.model.eval()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            outputs = self.model(X_torch)
        return outputs.cpu().numpy()

    def save(self, path: Path) -> None:
        """
        Save the model state dict and the hyperparameters.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Fit or load a model first.")

        checkpoint = {
            'state_dict': self.model.state_dict(),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'top_k': self.top_k,
            'num_kernels': self.num_kernels,
            'e_layers': self.e_layers,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'c_out': self.c_out,
            'enc_in': self.enc_in,
            'freq': self.freq,
            'embed': self.embed,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'target_indexes': self.model.target_indexes,
        }
        torch.save(checkpoint, str(path))

    def load(self, path: Path) -> None:
        """
        Load model weights and hyperparameters from disk.
        """
        checkpoint = torch.load(str(path), map_location=self._device)
        self.seq_len = checkpoint['seq_len']
        self.pred_len = checkpoint['pred_len']
        self.top_k = checkpoint['top_k']
        self.num_kernels = checkpoint['num_kernels']
        self.e_layers = checkpoint['e_layers']
        self.d_model = checkpoint['d_model']
        self.d_ff = checkpoint['d_ff']
        self.c_out = checkpoint['c_out']
        self.enc_in = checkpoint['enc_in']
        self.freq = checkpoint['freq']
        self.embed = checkpoint['embed']
        self.dropout = checkpoint['dropout']
        self.learning_rate = checkpoint['learning_rate']
        target_indexes = checkpoint['target_indexes']

        self.model = self._build_model(target_indexes)
        self.model.load_state_dict(checkpoint['state_dict'])

    @staticmethod
    def sample(trial: Trial, seq_len: int) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna or other tuning.
        Adjust the search space to your preference.
        """
        top_k_max = min(5, max(1, seq_len // 2))
        top_k = trial.suggest_int('top_k', 1, top_k_max)
        num_kernels = trial.suggest_int('num_kernels', 1, 6)
        e_layers = trial.suggest_int('e_layers', 1, 4)
        d_model = trial.suggest_int('d_model', 16, 128, step=16)
        d_ff = trial.suggest_int('d_ff', 16, 256, step=16)
        dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.1)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

        return {
            'top_k': top_k,
            'num_kernels': num_kernels,
            'e_layers': e_layers,
            'd_model': d_model,
            'd_ff': d_ff,
            'dropout': dropout,
            'learning_rate': learning_rate,
        }

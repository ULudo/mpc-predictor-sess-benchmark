
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple
from optuna import Trial

from .base_predictor import BasePredictor
from .helpers import train_and_evaluate_model, gen_data_loader
from ...util.consts_and_types import PredictionModelSets

from .net.time_mixer import TimeMixerModel  # Example path if your code is in net/time_mixer.py


class TimeMixerPredictor(BasePredictor):
    """
    A predictor that uses the TimeMixerModel for time-series forecasting.
    """

    def __init__(
        self,
        seq_len: int = 96,
        label_len: int = 0,
        pred_len: int = 96,
        down_sampling_window: int = 2,
        channel_independence: bool = False,
        e_layers: int = 2,
        moving_avg: int = 25,
        enc_in: int = 8,
        d_model: int = 32,
        embed: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1,
        d_ff: int = 64,
        c_out: int = 1,
        use_norm: int = 1,
        down_sampling_layers: int = 1,
        down_sampling_method: str = 'avg',
        top_k: int = 5,
        decomp_method: str = 'moving_avg',
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        early_stopping: bool = False,
        patience: int = 10,
        delta: float = 0.0,
        checkpoint_path: str = 'checkpoint.pt',
        noise_std: float = 0.0,
        scale_std: float = 1.0,
        device: str = 'cpu',
        verbose: bool = True,
        **kwargs
    ):
        """
        Args:
            seq_len (int): Input sequence length.
            label_len (int): If using partial encoder-decoder style, else 0.
            pred_len (int): Forecast horizon (# of future time steps to predict).
            down_sampling_window (int): Factor for down-sampling in TimeMixer.
            channel_independence (bool): If True, treat each channel separately.
            e_layers (int): Number of PastDecomposableMixing blocks.
            moving_avg (int): Kernel size for moving-average decomposition (if used).
            enc_in (int): Number of input features.
            d_model (int): Dimension of hidden embeddings in TimeMixer.
            embed (str): 'fixed' or 'timeF' or similar (embedding type).
            freq (str): Frequency code (e.g. 'h' or 't') for embeddings.
            dropout (float): Dropout probability.
            d_ff (int): Hidden dimension in feed-forward layers.
            c_out (int): Number of output features (1 if just predicting load).
            use_norm (int): Whether to use normalization layers.
            down_sampling_layers (int): How many times we down-sample.
            down_sampling_method (str): 'avg', 'max', or 'conv' for down-sampling.
            top_k (int): For DFT-based decomposition, how many freqs to keep.
            decomp_method (str): 'moving_avg' or 'dft_decomp'.
            learning_rate (float): Optimizer learning rate.
            batch_size (int): Mini-batch size.
            epochs (int): Number of training epochs.
            device (str): 'cpu' or 'cuda:<index>'.
            verbose (bool): Whether to print training/validation loss each epoch.
            kwargs: Additional unused kwargs for compatibility.
        """
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence
        self.e_layers = e_layers
        self.moving_avg = moving_avg
        self.enc_in = enc_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.d_ff = d_ff
        self.c_out = c_out
        self.use_norm = use_norm
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.top_k = top_k
        self.decomp_method = decomp_method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.noise_std = noise_std
        self.scale_std = scale_std
        self.verbose = verbose
        self._device = torch.device(device)
        self.model: nn.Module = None

    def _build_model(self, target_indexes:np.ndarray) -> nn.Module:
        """
        Build and return a TimeMixerModel instance on the correct device.
        """
        model = TimeMixerModel(
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            down_sampling_window=self.down_sampling_window,
            channel_independence=self.channel_independence,
            e_layers=self.e_layers,
            moving_avg=self.moving_avg,
            enc_in=self.enc_in,
            d_model=self.d_model,
            embed=self.embed,
            freq=self.freq,
            dropout=self.dropout,
            d_ff=self.d_ff,
            c_out=self.c_out,
            use_norm=self.use_norm,
            down_sampling_layers=self.down_sampling_layers,
            down_sampling_method=self.down_sampling_method,
            top_k=self.top_k,
            decomp_method=self.decomp_method,
            target_indexes=target_indexes
        )
        return model.to(self._device)

    def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the TimeMixer model for 'epochs' epochs on the given data.

        data.X_train: (N_train, seq_len, enc_in)
        data.y_train: (N_train, pred_len[, c_out])
        data.X_val:   (N_val, seq_len, enc_in) [optional]
        data.y_val:   (N_val, pred_len[, c_out]) [optional]

        Returns: (train_losses, val_losses)
        """
        if self.model is None:
            self.model = self._build_model(data.target_indexes)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_losses, val_losses = train_and_evaluate_model(
            model=self.model,
            data=data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            optimizer=optimizer,
            criterion=criterion,
            early_stopping=self.early_stopping,
            patience=self.patience,
            delta=self.delta,
            checkpoint_path=self.checkpoint_path,
            noise_std=self.noise_std,
            scale_std=self.scale_std,
            verbose=self.verbose,
            device=self._device,
        )
        return train_losses, val_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the next 'pred_len' steps.
        X shape: (N, seq_len, enc_in).

        Returns:
            np.ndarray of shape (N, pred_len, c_out) or (N, pred_len) if c_out=1.
        """
        if self.model is None:
            raise RuntimeError("TimeMixerPredictor has not been fitted yet.")

        self.model.eval()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            # This model signature is typically: forward(x_enc, x_mark_enc).
            # If you have no time-mark data, pass None for x_mark_enc.
            outputs = self.model(X_torch, None)  # => shape: (N, pred_len, c_out)
        return outputs.cpu().numpy()

    def save(self, path: Path) -> None:
        """
        Save model state and hyperparameters to disk.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Fit or load a model first.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'down_sampling_window': self.down_sampling_window,
            'channel_independence': self.channel_independence,
            'e_layers': self.e_layers,
            'moving_avg': self.moving_avg,
            'enc_in': self.enc_in,
            'd_model': self.d_model,
            'embed': self.embed,
            'freq': self.freq,
            'dropout': self.dropout,
            'd_ff': self.d_ff,
            'c_out': self.c_out,
            'use_norm': self.use_norm,
            'down_sampling_layers': self.down_sampling_layers,
            'down_sampling_method': self.down_sampling_method,
            'top_k': self.top_k,
            'decomp_method': self.decomp_method,
            'learning_rate': self.learning_rate,
            'target_indexes': self.model.target_indexes,
        }
        torch.save(checkpoint, str(path))

    def load(self, path: Path) -> None:
        """
        Load model state and hyperparameters from disk.
        """
        checkpoint = torch.load(str(path), map_location=self._device)

        self.seq_len = checkpoint['seq_len']
        self.label_len = checkpoint['label_len']
        self.pred_len = checkpoint['pred_len']
        self.down_sampling_window = checkpoint['down_sampling_window']
        self.channel_independence = checkpoint['channel_independence']
        self.e_layers = checkpoint['e_layers']
        self.moving_avg = checkpoint['moving_avg']
        self.enc_in = checkpoint['enc_in']
        self.d_model = checkpoint['d_model']
        self.embed = checkpoint['embed']
        self.freq = checkpoint['freq']
        self.dropout = checkpoint['dropout']
        self.d_ff = checkpoint['d_ff']
        self.c_out = checkpoint['c_out']
        self.use_norm = checkpoint['use_norm']
        self.down_sampling_layers = checkpoint['down_sampling_layers']
        self.down_sampling_method = checkpoint['down_sampling_method']
        self.top_k = checkpoint['top_k']
        self.decomp_method = checkpoint['decomp_method']
        self.learning_rate = checkpoint['learning_rate']
        target_indexes = checkpoint['target_indexes']

        # Rebuild and load
        self.model = self._build_model(target_indexes)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    @staticmethod
    def sample(trial: Trial, seq_len:int) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna. Modify as desired.
        """
        e_layers = trial.suggest_int('e_layers', 1, 3)
        if seq_len <= 2:
            down_sampling_window = 1
        else:
            max_down_sampling_window = min(4, seq_len - 1)
            down_sampling_window = trial.suggest_int('down_sampling_window', 2, max_down_sampling_window)
        down_sampling_layers = trial.suggest_int('down_sampling_layers', 1, 2)

        # Compute the minimal final length after down-sampling (e.g. 96 -> 48 -> 24 for (window=2, layers=2))
        final_len = seq_len // (down_sampling_window ** down_sampling_layers)
        # just to avoid going below 1
        final_len = max(1, final_len)
        max_moving_avg = min(25, final_len)
        moving_avg = trial.suggest_int("moving_avg", 1, max_moving_avg, step=2)  # only odds: 1,3,5,7,...

        max_topk = min(5, max(1, seq_len // 2))
        d_model = trial.suggest_int('d_model', 16, 64, step=16)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        d_ff = trial.suggest_int('d_ff', 16, 128, step=16)
        top_k = trial.suggest_int('top_k', 1, max_topk)
        decomp_method = trial.suggest_int('decomp_method', 0, 1)
        decomp_method = 'moving_avg' if decomp_method == 0 else 'dft_decomp'
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        return {
            'down_sampling_window': down_sampling_window,
            'e_layers': e_layers,
            'moving_avg': moving_avg,
            'd_model': d_model,
            'dropout': dropout,
            'd_ff': d_ff,
            'down_sampling_layers': down_sampling_layers,
            'top_k': top_k,
            'decomp_method': decomp_method,
            'learning_rate': learning_rate,
        }
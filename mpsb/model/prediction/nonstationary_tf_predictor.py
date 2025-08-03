import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple
from optuna import Trial

from .base_predictor import BasePredictor
from ...util.consts_and_types import PredictionModelSets
from .helpers import train_and_evaluate_model, gen_data_loader

from .net.nonstationary_tf import NonstationaryTf  # Adjust the import path as needed


class NonstationaryTfPredictor(BasePredictor):
    """
    A predictor that uses the Nonstationary Transformer (NonstationaryTf)
    for time-series forecasting.
    """

    def __init__(
        self,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        enc_in: int = 8,
        dec_in: int = 8,
        d_model: int = 64,
        embed: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1,
        factor: int = 5,
        n_heads: int = 8,
        d_ff: int = 256,
        activation: str = 'relu',
        e_layers: int = 2,
        d_layers: int = 1,
        c_out: int = 1,
        p_hidden_dims=(64, 64),
        p_hidden_layers=2,
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
            label_len (int): portion of seq_len used as "known" in the decoder.
            pred_len (int): forecast horizon (# of future steps).
            enc_in (int): number of input features in the encoder.
            dec_in (int): number of input features in the decoder.
            d_model (int): dimension of the internal embeddings.
            embed (str): embedding type, e.g. 'fixed'.
            freq (str): data frequency code, e.g. 'h' or 't'.
            dropout (float): dropout probability.
            factor (int): factor for the DSAttention.
            n_heads (int): number of attention heads.
            d_ff (int): dimension of the feedforward layers.
            activation (str): 'relu' or 'gelu'.
            e_layers (int): how many encoder layers.
            d_layers (int): how many decoder layers.
            c_out (int): number of output channels (1 if you only predict one).
            p_hidden_dims (tuple): hidden dims for the tau/delta MLPs.
            p_hidden_layers (int): how many linear + activation blocks in the tau/delta MLPs.
            learning_rate (float): optimizer LR.
            batch_size (int): training mini-batch size.
            epochs (int): training epochs.
            device (str): 'cpu' or 'cuda:0' etc.
            verbose (bool): whether to print epoch losses.
            kwargs: additional unused arguments for compatibility.
        """
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.c_out = c_out
        self.p_hidden_dims = p_hidden_dims
        self.p_hidden_layers = p_hidden_layers

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

    def _build_model(self, target_indexes:np.ndarray) -> nn.Module:
        """
        Create an instance of NonstationaryTf on the appropriate device.
        """
        model = NonstationaryTf(
            pred_len=self.pred_len,
            seq_len=self.seq_len,
            label_len=self.label_len,
            enc_in=self.enc_in,
            dec_in=self.dec_in,
            d_model=self.d_model,
            embed=self.embed,
            freq=self.freq,
            dropout=self.dropout,
            factor=self.factor,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            activation=self.activation,
            e_layers=self.e_layers,
            c_out=self.c_out,
            d_layers=self.d_layers,
            p_hidden_dims=self.p_hidden_dims,
            p_hidden_layers=self.p_hidden_layers,
            target_indexes=target_indexes
        )
        return model.to(self._device)

    def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the Nonstationary Transformer.

        data.X_train: shape (N_train, seq_len, enc_in)
        data.y_train: shape (N_train, pred_len, c_out) or (N_train, pred_len)
        data.X_val: shape (N_val, seq_len, enc_in) [optional]
        data.y_val: shape (N_val, pred_len, c_out) [optional]

        Returns:
            (train_losses, val_losses)
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
            early_stopping = self.early_stopping,
            patience = self.patience,
            delta = self.delta,
            checkpoint_path = self.checkpoint_path,
            verbose=self.verbose
        )
        return train_losses, val_losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the next 'pred_len' steps.
        X: shape (N, seq_len, enc_in)

        We pass x_enc=X, x_mark_enc=None for simplicity.
        We'll create x_dec as a placeholder (since NonstationaryTf forward expects x_dec).

        Returns:
            np.ndarray of shape (N, pred_len[, c_out]).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")

        self.model.eval()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            outputs = self.model(x_enc=X_torch)
        return outputs.cpu().numpy()

    def save(self, path: Path) -> None:
        """
        Save model weights & hyperparameters.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Fit or load first.")

        checkpoint = {
            'state_dict': self.model.state_dict(),
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'enc_in': self.enc_in,
            'dec_in': self.dec_in,
            'd_model': self.d_model,
            'embed': self.embed,
            'freq': self.freq,
            'dropout': self.dropout,
            'factor': self.factor,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'activation': self.activation,
            'e_layers': self.e_layers,
            'd_layers': self.d_layers,
            'c_out': self.c_out,
            'p_hidden_dims': self.p_hidden_dims,
            'p_hidden_layers': self.p_hidden_layers,
            'target_indexes': self.model.target_indexes,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': self.verbose
        }
        torch.save(checkpoint, str(path))

    def load(self, path: Path) -> None:
        """
        Load model weights & hyperparameters from a saved file.
        """
        checkpoint = torch.load(str(path), map_location=self._device)

        self.seq_len = checkpoint['seq_len']
        self.label_len = checkpoint['label_len']
        self.pred_len = checkpoint['pred_len']
        self.enc_in = checkpoint['enc_in']
        self.dec_in = checkpoint['dec_in']
        self.d_model = checkpoint['d_model']
        self.embed = checkpoint['embed']
        self.freq = checkpoint['freq']
        self.dropout = checkpoint['dropout']
        self.factor = checkpoint['factor']
        self.n_heads = checkpoint['n_heads']
        self.d_ff = checkpoint['d_ff']
        self.activation = checkpoint['activation']
        self.e_layers = checkpoint['e_layers']
        self.d_layers = checkpoint['d_layers']
        self.c_out = checkpoint['c_out']
        self.p_hidden_dims = checkpoint['p_hidden_dims']
        self.p_hidden_layers = checkpoint['p_hidden_layers']
        target_indexes = checkpoint['target_indexes']

        self.learning_rate = checkpoint['learning_rate']
        self.batch_size = checkpoint['batch_size']
        self.epochs = checkpoint['epochs']
        self.verbose = checkpoint['verbose']

        self.model = self._build_model(target_indexes)
        self.model.load_state_dict(checkpoint['state_dict'])

    @staticmethod
    def sample(trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Optuna or other tuning.
        Adjust as you see fit. Some params can be fixed externally.
        """
        d_model = trial.suggest_int('d_model', 32, 64, step=32)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        factor = trial.suggest_int('factor', 3, 5)
        n_heads = trial.suggest_int('n_heads', 2, 4, step=2)
        d_ff = trial.suggest_int('d_ff', 64, 128, step=64)
        activation = trial.suggest_categorical('activation', ['relu', 'gelu'])
        e_layers = trial.suggest_int('e_layers', 2, 4)
        d_layers = trial.suggest_int('d_layers', 1, 2)
        p_hidden_dims = trial.suggest_categorical ('p_hidden_dims', [(32, 32), (64, 64), (128, 128)])
        p_hidden_layers = trial.suggest_int('p_hidden_layers', 1, 3)

        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

        return {
            'd_model': d_model,
            'dropout': dropout,
            'factor': factor,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'activation': activation,
            'e_layers': e_layers,
            'd_layers': d_layers,
            'p_hidden_dims': p_hidden_dims,
            'p_hidden_layers': p_hidden_layers,
            'learning_rate': learning_rate,
        }
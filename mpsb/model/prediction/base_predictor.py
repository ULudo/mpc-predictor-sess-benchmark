import abc
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
from optuna import Trial

from mpsb.util.consts_and_types import PredictionModelSets


class BasePredictor(object):
        """
        Abstract base class for prediction models.
        """
        def __init__(self, **kwargs) -> None:
            pass

        @abc.abstractmethod
        def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
            """
            Fit the model to the data.
            :param data: A dictionary containing the training data.
            :return: A tuple containing evaluation results for the training and validation sets.
            """
            pass

        @abc.abstractmethod
        def predict(self, X: np.ndarray) -> np.ndarray:
            pass

        def predict_in_batches(self, X: np.ndarray, batch_size:int) -> np.ndarray:
            y_pred = []
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch_pred = self.predict(X_batch)
                if y_batch_pred.ndim == 1:
                    y_batch_pred = y_batch_pred.reshape(1, -1)
                y_pred.append(y_batch_pred)
            return np.concatenate(y_pred, axis=0)

        @abc.abstractmethod
        def save(self, path: Path) -> None:
            pass

        @abc.abstractmethod
        def load(self, path: Path) -> None:
            pass

        @staticmethod
        @abc.abstractmethod
        def sample(trial:Trial) -> Dict[str, Any]:
            pass
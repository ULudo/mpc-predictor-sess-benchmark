from .base_predictor import BasePredictor
from typing import Dict, Any, Tuple
from typing import Union
import numpy as np
import pandas as pd
from optuna import Trial
from sklearn.ensemble import RandomForestRegressor
import joblib

from ...util.consts_and_types import PredictionModelSets


class RandomForestPredictor(BasePredictor):
    def __init__(self, **kwargs):
        """
        Initialize a random forest regressor.
        kwargs are passed to sklearn's RandomForestRegressor.
        """
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, data:PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        self.model.fit(data.X_train, data.y_train)
        y_train_pred = self.model.predict(data.X_train)
        y_val_pred = self.model.predict(data.X_val)
        return y_train_pred, y_val_pred

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)

    @staticmethod
    def sample(trial: Trial) -> Dict[str, Any]:
        return {
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
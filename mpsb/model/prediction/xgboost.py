from typing import Dict, Any, Tuple

import joblib
import numpy as np
from optuna import Trial
from xgboost import XGBRegressor

from .base_predictor import BasePredictor
from ...util.consts_and_types import PredictionModelSets


class XGBoostPredictor(BasePredictor):
    def __init__(self, **kwargs):
        """
        Initialize an XGBoost regressor.
        kwargs are passed directly to XGBRegressor.
        """
        super().__init__(**kwargs)
        self.eval_metric = kwargs.get('eval_metric', 'rmse')
        self.model = XGBRegressor(**kwargs)
        self.kwargs = kwargs

    def _shuffle(self, X, y):
        size = X.shape[0]
        idx = np.random.permutation(size)
        return X[idx], y[idx]

    def fit(self, data: PredictionModelSets) -> Tuple[np.ndarray, np.ndarray]:
        X_train, y_train = self._shuffle(data.X_train, data.y_train)
        eval_set = [(X_train, y_train)]
        if data.X_val is not None:
            X_val, y_val = self._shuffle(data.X_val, data.y_val)
            eval_set.append((X_val, y_val))
        self.model.fit(X_train, y_train, eval_set=eval_set, xgb_model=self.model)
        res = self.model.evals_result()
        eval_train = np.array(res['validation_0'][self.eval_metric])
        if data.X_val is None:
            eval_val = np.array([])
        else:
            eval_val = np.array(res['validation_1'][self.eval_metric])
        return eval_train, eval_val

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)

        for key, value in self.kwargs.items():
            setattr(self.model, key, value)

    @staticmethod
    def sample(trial: Trial) -> Dict[str, Any]:
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Tree depth
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),  # Step size
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),  # Row sampling
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),  # Feature sampling
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  # L1 regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),  # L2 regularization
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),  # Min sum of instance weight
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),  # Min loss reduction for split
        }

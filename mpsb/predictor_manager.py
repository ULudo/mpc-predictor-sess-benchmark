import os
from enum import Enum
from pathlib import Path
from typing import Union, List, Tuple, Any, Dict, Type, Optional

import joblib
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from mpsb.env import BuildingDataManager
from mpsb.model.prediction.base_predictor import BasePredictor
from mpsb.model.prediction.data_preprocessor import DataPreprocessor, FeatureExtractor
from mpsb.model.prediction.evaluation import evaluate_prediction_model
from mpsb.model.prediction.linear_predictor import LinearPredictor
from mpsb.model.prediction.nonstationary_tf_predictor import NonstationaryTfPredictor
from mpsb.model.prediction.predictor_tuner import PredictorTuner
from mpsb.model.prediction.time_mixer_predictor import TimeMixerPredictor
from mpsb.model.prediction.times_net_predictor import TimesNetPredictor
from mpsb.util.consts_and_types import PredictionModelSets,  DataColumn
from mpsb.model.prediction.random_forest import RandomForestPredictor
from mpsb.model.prediction.recurrent_net import RecurrentPredictor
from mpsb.model.prediction.xgboost import XGBoostPredictor


class PredictorType(Enum):
    RECURRENT_NET = RecurrentPredictor
    XGBOOST = XGBoostPredictor
    RANDOM_FOREST = RandomForestPredictor
    TIMES_NET = TimesNetPredictor
    TIME_MIXER = TimeMixerPredictor
    NONSTATIONARY_TF = NonstationaryTfPredictor
    LINEAR = LinearPredictor


class PredictorManager:

    def __init__(self, config: DictConfig, log_dir: Path):
        self.log_dir = log_dir
        self.config = config
        self.data = None

    def train(self) -> BasePredictor:
        self._ensure_data_is_set()
        predictor = self.create_predictor()
        res_eval_train, res_eval_val = predictor.fit(self.data)
        np.save(os.path.join(self.log_dir, f'fit_eval_train.npy'), res_eval_train)
        np.save(os.path.join(self.log_dir, f'fit_eval_val.npy'), res_eval_val)
        predictor.save(self.log_dir / 'predictor.pkl')
        return predictor

    def create_predictor(self) -> BasePredictor:
        kwargs = self.config.model.get('kwargs', {})
        if kwargs.get('early_stopping', False):
            kwargs["checkpoint_path"] = str(self.log_dir / 'model_checkpoint.pt')
        predictor = self.get_predictor(self.config.model.predictor_type)(**kwargs)
        return predictor

    def load_predictor(self) -> BasePredictor:
        model_dir = self.config.model.get("model_dir")
        predictor = self.create_predictor()
        predictor.load(model_dir)
        return predictor

    def tune(self) -> None:
        self._ensure_data_is_set()
        tuner = PredictorTuner(
            log_dir=self.log_dir,
            predictor=self.get_predictor(self.config.model.predictor_type),
            data=self.data,
            n_trials=self.config.model.get('n_trials'),
            sampler=self.config.model.get('sampler', 'tpe'),
            n_startup_trials=self.config.model.get('n_startup_trials', 10),
            pruner=self.config.model.get('pruner', 'median'),
            n_jobs=self.config.model.get('n_jobs', 1),
            study_name=self.config.model.predictor_type,
            fix_kwargs=self.config.model.get('kwargs', {}),
            n_gpus=self.config.model.get('n_gpus', 0),
            sampler_kwargs=self.config.model.get('sampler_kwargs', {}),
            seed=self.config.get('seed', None),
        )
        tuner.tune()

    def evaluate(self, predictor: BasePredictor) -> None:
        self._ensure_data_is_set()
        batch_size = self.config.model.get("kwargs", {}).get("batch_size", 0)
        names = ['train', 'validation', 'test']
        input_sets = [self.data.X_train, self.data.X_val, self.data.X_test]
        output_sets = [self.data.y_train, self.data.y_val, self.data.y_test]
        results = []
        for name, X, y in zip(names, input_sets, output_sets):
            if X is not None:
                if batch_size:
                    y_pred = predictor.predict_in_batches(X, batch_size)
                else:
                    y_pred = predictor.predict(X)
                eval_results = evaluate_prediction_model(y, y_pred)
                results.append({'set': name, **eval_results})
        # save results as csv
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.log_dir / 'evaluation_results.csv', index=False)

    def train_and_evaluate(self) -> None:
        predictor = self.train()
        self.evaluate(predictor)

    def prepare_data(self, two_d: bool = False) -> None:
        data_class, data_mgr_dfs, feature_extractor = self._determine_predictor_data()

        preprocessor = DataPreprocessor(
            seq_length=self.config.model.sequence_length,
            prediction_steps=self.config.model.prediction_steps
        )

        def _prepare_check_data(
                periods: Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if periods:
                return preprocessor.prepare_evaluation_data(
                    data_mgr_dfs=data_mgr_dfs,
                    evaluation_periods=periods,
                    feature_extractor=feature_extractor,
                    scaler=scaler,
                    two_d=two_d,
                )
            return None, None, None

        data_check_defs = self.config.data.get(data_class, None)
        eval_periods = self._get_check_periods(data_check_defs, "eval") if data_check_defs else {}
        test_periods = self._get_check_periods(data_check_defs, "test") if data_check_defs else {}
        X_train, y_train, target_indexes, scaler, n_orig_data_columns = preprocessor.prepare_train_data(
            data_mgr_dfs=data_mgr_dfs,
            eval_periods=eval_periods,
            test_periods=test_periods,
            feature_extractor=feature_extractor,
            two_d=two_d,
        )
        X_val, y_val, _ = _prepare_check_data(eval_periods)
        X_test, y_test, _ = _prepare_check_data(test_periods)

        aug_cols = [i for i in range(n_orig_data_columns)] \
            if self.config.model.get('data_augmentation', False) else []

        self.data = PredictionModelSets(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            target_indexes=target_indexes,
            scaler=scaler,
            aug_cols=aug_cols,
        )

    def save_scaler(self) -> None:
        self._ensure_data_is_set()
        joblib.dump(self.data.scaler, self.log_dir / 'scaler.pkl')

    @staticmethod
    def load_scaler(path:str) -> StandardScaler:
        return joblib.load(path)

    def _determine_predictor_data(self) -> Tuple[str, Dict[str, pd.DataFrame], FeatureExtractor]:
        label = DataColumn[self.config.data.label.upper()]
        if label in [DataColumn.LOAD, DataColumn.PV]:
            feature_extractor = self._determine_feature_extractor(BuildingDataManager.building_cols, label)
            return "building", BuildingDataManager.building_dfs, feature_extractor
        elif label == DataColumn.PRICE:
            feature_extractor = self._determine_feature_extractor(BuildingDataManager.price_cols, label)
            return "price", BuildingDataManager.price_dfs, feature_extractor

    def _determine_feature_extractor(self, default: List[str], label: DataColumn) -> FeatureExtractor:
        features = self.config.data.get('features', default) # TODO: might convert to data column
        feature_extractor = FeatureExtractor(
            features=features,
            labels=[label],
            use_time_features=self.config.model.get('use_time_features', True),
            sin_cos_encoding=self.config.model.get('sin_cos_encoding', True),
        )
        return feature_extractor

    def _ensure_data_is_set(self):
        if self.data is None:
            raise ValueError("Data is not set. Please call prepare_data() before proceeding.")

    @staticmethod
    def get_predictor(name:str) -> Type[BasePredictor]:
        return PredictorType[name.upper()].value

    @staticmethod
    def _get_check_periods(
            conf: Any,
            check_set: str
    ) -> Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]:
        get_periods = lambda item: [(i.start_date, i.end_date) for i in getattr(item, "periods")]
        if data_sets := getattr(conf, check_set, None):
            return {d.name: get_periods(d) for d in data_sets}
        return {}

from typing import Tuple, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model
from sklearn.preprocessing import StandardScaler

from mpsb.model.base_controller import BaseController
from mpsb.model.prediction.base_predictor import BasePredictor
from mpsb.util.consts_and_types import DATA_FREQUENCY, InfoMsg, PredictionModelSets
from mpsb.util.functions import s_to_hour

@dataclass
class PredictorBundle:
    predictor: BasePredictor
    scaler: StandardScaler
    scaler_target_idx: int
    two_d: bool

class MPCOptimizer:

    def __init__(
            self,
            n_predictions: int,
            bat_efficiency: float,
            bat_capacity: float,
            bat_max_power: float,
            tax: float = 0.0,
    ) -> None:
        self.dt = s_to_hour(DATA_FREQUENCY)
        self.n_predictions = n_predictions
        self.bat_efficiency = bat_efficiency
        self.bat_capacity = bat_capacity  # in Wh
        self.bat_max_power = bat_max_power  # in W
        self.tax = tax

    def optimize(
            self,
            soc_initial: float,
            load_ts: np.ndarray,  # in kWh
            price_ts: np.ndarray,  # in €/kWh
            pv_ts: np.ndarray  # in kWh
    ) -> Tuple[float, float]:
        time_steps = self.n_predictions + 1  # Include current time step

        # Create a linopy model
        m = Model()

        # Define time indices
        time = pd.Index(range(time_steps), name='time')
        # Defined for time steps t = 0 to N (total N+1 steps) because you need to know the SOC after the last action at time t = N-1.
        # SOC[0] -> p_charge/p_discharge[0] -> SOC[1] -> p_charge/p_discharge[1] -> ... -> SOC[N-1] -> p_charge/p_discharge[N-1] -> SOC[N]
        time_soc = pd.Index(range(time_steps + 1), name='time')
        # Define coords for variables
        coords = {'time': time}
        coords_soc = {'time': time_soc}
        # Define variables
        soc = m.add_variables(name="soc", dims=["time"], coords=coords_soc, lower=0, upper=1)
        p_charge = m.add_variables(name="p_charge", dims=["time"], coords=coords, lower=0,
                                   upper=self.bat_max_power)  # in W
        p_discharge = m.add_variables(name="p_discharge", dims=["time"], coords=coords, lower=0,
                                      upper=self.bat_max_power)  # in W

        # Convert prediction arrays to xarrays with time coordinate for the use with linopy
        load_da = xr.DataArray(load_ts, dims=['time'], coords=coords)  # in kWh
        price_da = xr.DataArray(price_ts, dims=['time'], coords=coords)  # in €/kWh
        pv_da = xr.DataArray(pv_ts, dims=['time'], coords=coords)  # in kWh

        # Convert p_charge and p_discharge from W to kW
        p_charge_kwh = p_charge * self.dt / 1000  # W to kWh
        p_discharge_kwh = p_discharge * self.dt / 1000  # W to kWh

        # Compute net power (kWh)
        net_power = p_charge_kwh - p_discharge_kwh + load_da - pv_da

        # Add variables for net import and net export
        net_import = m.add_variables(name="net_import", dims=["time"], coords=coords, lower=0)
        net_export = m.add_variables(name="net_export", dims=["time"], coords=coords, lower=0)

        # Price for importing electricity
        price_import = price_da + self.tax

        # Constraints
        # Initial SoC constraint
        # This constraint sets the initial SoC of the battery at the beginning of the prediction horizon to match the actual current SoC.
        # Ensures that the optimization model starts from the correct SoC, reflecting the real state of the battery.
        m.add_constraints(soc.sel(time=0) - soc_initial == 0, name="initial_soc")
        # SoC dynamics constraint
        # This constraint models the battery's SoC dynamics over the prediction horizon.
        # It enforces the relationship between the SoC at consecutive time steps based on the charging and discharging actions.
        soc_diff = soc.sel(time=time + 1) - soc.sel(time=time) - (
                (
                            p_charge * self.dt * self.bat_efficiency - p_discharge * self.dt / self.bat_efficiency) / self.bat_capacity
        )
        m.add_constraints(soc_diff == 0, name="soc_dynamics")
        # Net power balance constraint
        # The constraint is essential to correctly model the net import and export of electricity to and
        # from the grid and to apply the tax only when importing electricity.
        m.add_constraints(net_power == net_import - net_export, name="net_power_balance")

        # Objective: Minimize total cost
        cost = (price_import * net_import - price_da * net_export).sum(dim="time")
        m.objective = cost

        # Solve the optimization problem
        result = m.solve(solver_name="gurobi")

        # Check if the optimization was successful
        if result != ('ok', 'optimal'):
            raise RuntimeError(f"Optimization failed with status: {result}")

        # Extract the optimal p_charge and p_discharge at time=0
        p_charge_opt = p_charge.solution.sel(time=0).item()
        p_discharge_opt = p_discharge.solution.sel(time=0).item()

        return p_charge_opt, p_discharge_opt

    def get_action(self, p_charge_opt, p_discharge_opt):
        # Compute net power (positive for charging, negative for discharging)
        p_net = p_charge_opt - p_discharge_opt  # in W

        # Map net power to action space [-1, 1]
        action = p_net / self.bat_max_power
        assert -1.0 <= action <= 1.0

        return action


class PerfectMPController(BaseController):

    def __init__(self, optimizer:MPCOptimizer) -> None:
        self.optimizer = optimizer

    def step(self, obs: np.ndarray, info: InfoMsg) -> int | float | np.ndarray:
        opti_params = self._get_params_from_info(info)
        p_charge_opt, p_discharge_opt = self.optimizer.optimize(*opti_params)
        return self.optimizer.get_action(p_charge_opt, p_discharge_opt)

    def _get_params_from_info(self, info) -> Any:
        obs_dict = info.observation
        soc = obs_dict["soc"]
        to_ts_array = lambda name: np.array([obs_dict[f"{name}_{i}"] for i in range(self.optimizer.n_predictions + 1)])
        opti_params = [soc] + [to_ts_array(name) for name in ["loads", "prices", "gens"]]
        return opti_params


class PredictorMPController(BaseController):
    """
    An MPC controller that uses imperfect (learned) predictions for MPC.
    """

    def __init__(
            self,
            load_predictor: PredictorBundle,
            pv_predictor: PredictorBundle,
            price_predictor: PredictorBundle,
            optimizer: MPCOptimizer,
            history_length: int,
    ) -> None:
        self.optimizer = optimizer
        self.load_predictor = load_predictor
        self.pv_predictor = pv_predictor
        self.price_predictor = price_predictor
        self.history_length = history_length

        self._load_history = np.zeros(history_length, dtype=np.float32)
        self._pv_history = np.zeros(history_length, dtype=np.float32)
        self._price_history = np.zeros(history_length, dtype=np.float32)
        self._time_feat_history = np.zeros((history_length, 6), dtype=np.float32)
        self._last_forecasts = None
        self._last_truth = None

    def step(self, obs: np.ndarray, info: InfoMsg) -> int | float | np.ndarray:
        cur_load, cur_price, cur_pv, soc, time_feats = self.extract_information_from_info(info)
        self._update_histories(cur_load, cur_price, cur_pv, time_feats)
        forecast_load, forecast_price, forecast_pv = self._make_predictions(cur_load, cur_price, cur_pv)
        p_charge_opt, p_discharge_opt = self.optimizer.optimize(
            soc, forecast_load, forecast_price, forecast_pv
        )
        # store for logging
        self._last_truth = {"load": cur_load, "pv": cur_pv, "price": cur_price}
        self._last_forecasts = {"load": forecast_load, "pv": forecast_pv, "price": forecast_price}
        return  self.optimizer.get_action(p_charge_opt, p_discharge_opt)

    def _make_predictions(
        self, cur_load: float, cur_price: float, cur_pv: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictor_input_building = np.column_stack((self._load_history, self._pv_history, self._time_feat_history))
        predictor_input_price = np.column_stack((self._price_history, self._time_feat_history))
        forecast_load = self.prepare_forecast_for_optimization(self.load_predictor, predictor_input_building, cur_load)
        forecast_pv = self.prepare_forecast_for_optimization(self.pv_predictor, predictor_input_building, cur_pv)
        forecast_price = self.prepare_forecast_for_optimization(self.price_predictor, predictor_input_price, cur_price)
        return forecast_load, forecast_price, forecast_pv

    def _update_histories(
        self, cur_load: float, cur_price: float, cur_pv: float, time_feats: np.ndarray
    ) -> None:
        def _rolling_insert(arr, val):
            arr[:-1] = arr[1:]
            arr[-1] = val
        _rolling_insert(self._load_history, np.float32(cur_load))
        _rolling_insert(self._pv_history, np.float32(cur_pv))
        _rolling_insert(self._price_history, np.float32(cur_price))
        _rolling_insert(self._time_feat_history, time_feats.astype(np.float32))

    @staticmethod
    def extract_information_from_info(info: InfoMsg) -> Tuple[float, float, float, float, np.ndarray]:
        obs_dict = info.observation
        soc = obs_dict["soc"]
        cur_load = obs_dict["loads"]
        cur_pv = obs_dict["gens"]
        cur_price = obs_dict["prices"]
        time_feats = np.array([obs_dict[f"time_features_{i}"] for i in range(6)], dtype=np.float32)
        return cur_load, cur_price, cur_pv, soc, time_feats

    @staticmethod
    def prepare_forecast_for_optimization(
            predictor:PredictorBundle,
            data_stack:np.ndarray,
            cur_dp:float
    ) -> np.ndarray:
        predictor_input = PredictorMPController._scale_and_reshape_data(data_stack, predictor.scaler, predictor.two_d)
        predictions = predictor.predictor.predict(predictor_input)
        if predictor.scaler:
            predictions = PredictorMPController._unscale_prediction(predictor.scaler, predictions, predictor.scaler_target_idx)
        optimizer_input = np.concatenate(([cur_dp], predictions.squeeze()))
        return optimizer_input

    @staticmethod
    def _unscale_prediction(
            scaler:StandardScaler,
            predictions:np.ndarray,
            index:int=0
    ) -> np.ndarray:
        data_stack = np.zeros((np.prod(predictions.shape), scaler.scale_.shape[0]))
        data_stack[:, index] = predictions.flatten()
        return scaler.inverse_transform(data_stack)[:, index].reshape(predictions.shape)

    @staticmethod
    def _scale_and_reshape_data(
            data_stack: np.ndarray,
            scaler: Optional[StandardScaler],
            two_d: bool
    ) -> np.ndarray:
        if scaler:
            data_stack = scaler.transform(data_stack)

        if two_d:
            # shape: (history_length, num_features) -> (1, history_length * num_features)
            data_stack = data_stack.reshape(1, -1)
        else:
            # shape: (history_length, num_features) -> (1, history_length, num_features)
            data_stack = np.expand_dims(data_stack, axis=0)

        return data_stack


class RetrainPredictorMPController(BaseController):
    """
    An MPC controller that not only uses imperfect (learned) predictions for MPC
    but also retrains its predictors every retraining interval (e.g. every 24 hours).
    """

    def __init__(
            self,
            load_predictor: PredictorBundle,
            pv_predictor: PredictorBundle,
            price_predictor: PredictorBundle,
            optimizer:MPCOptimizer,
            history_length: int,
            prediction_horizon: int,
            retrain_interval: int  # e.g. 96 (i.e. retrain every 24 hours)
    ) -> None:
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.retrain_interval = retrain_interval
        self.buffer_size = history_length + prediction_horizon + retrain_interval
        # Buffer for measurements, 9-dimensional vectors: [load, pv, price, time_features (6)]
        self.buffer = np.zeros((self.buffer_size, 9), dtype=np.float32)
        self.buffer_filled = False
        self.n_steps = 0  # steps since last retraining

        self.load_predictor = load_predictor
        self.pv_predictor = pv_predictor
        self.price_predictor = price_predictor
        self.optimizer = optimizer
        self._last_forecasts = None
        self._last_truth = None

    def step(self, obs: np.ndarray, info: InfoMsg) -> int | float | np.ndarray:
        measurement, soc = self._extract_measurement_and_soc(info)
        self._update_rolling_buffer(measurement)
        forecast_load, forecast_price, forecast_pv = self._make_predictions(measurement)
        # store for logging
        self._last_truth = {"load": float(measurement[0]), "pv": float(measurement[1]), "price": float(measurement[2])}
        self._last_forecasts = {"load": forecast_load, "pv": forecast_pv, "price": forecast_price}
        p_charge_opt, p_discharge_opt = self.optimizer.optimize(soc, forecast_load, forecast_price, forecast_pv)
        action = self.optimizer.get_action(p_charge_opt, p_discharge_opt)
        return action

    @staticmethod
    def _extract_measurement_and_soc(info: InfoMsg) -> Tuple[np.ndarray, float]:
        """
        Extract the current state of charge (soc) and the measurement vector from the info message.
        The measurement vector is a 9-dimensional vector containing [load, pv, price, time_features (6 values)].
        """
        cur_load, cur_price, cur_pv, soc, time_feats = PredictorMPController.extract_information_from_info(info)
        measurement = np.concatenate((
            np.array([cur_load], dtype=np.float32),
            np.array([cur_pv], dtype=np.float32),
            np.array([cur_price], dtype=np.float32),
            time_feats.astype(np.float32)
        ), axis=0)
        return measurement, soc

    def _update_rolling_buffer(self, measurement: np.ndarray) -> None:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = measurement
        self.n_steps += 1
        if not self.buffer_filled and self.n_steps == self.buffer_size:
            self.buffer_filled = True
        if self.buffer_filled and self.n_steps >= self.retrain_interval:
            self._retrain_predictors()
            self.n_steps = 0

    def _make_predictions(
            self, measurement:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_slice = slice(-self.history_length, None)
        building_input = np.hstack((self.buffer[input_slice, [0, 1]], self.buffer[input_slice, 3:9]))
        price_input = np.hstack((self.buffer[input_slice, [2]], self.buffer[input_slice, 3:9]))
        forecast_load = PredictorMPController.prepare_forecast_for_optimization(
            self.load_predictor, building_input, float(measurement[0]))
        forecast_pv = PredictorMPController.prepare_forecast_for_optimization(
            self.pv_predictor, building_input, float(measurement[1]))
        forecast_price = PredictorMPController.prepare_forecast_for_optimization(
            self.price_predictor, price_input, float(measurement[2]))
        return forecast_load, forecast_price, forecast_pv

    def _retrain_predictors(self) -> None:
        # Measurement separation
        building_data = np.hstack((self.buffer[:, [0, 1]], self.buffer[:, 3:9]))
        price_data = np.hstack((self.buffer[:, [2]], self.buffer[:, 3:9]))
        assert building_data.shape[1] == 8, f"Expected 8 features in building data, got {building_data.shape[1]}"
        assert price_data.shape[1] == 7, f"Expected 7 features in price data, got {price_data.shape[1]}"

        # Data preparation
        data_load = self._prepare_training_data(building_data, 0, self.load_predictor.scaler, self.load_predictor.two_d)
        data_pv = self._prepare_training_data(building_data, 1, self.pv_predictor.scaler, self.pv_predictor.two_d)
        data_price = self._prepare_training_data(price_data, 0, self.price_predictor.scaler, self.price_predictor.two_d)

        # Retraining
        self._retrain_and_log("Load", self.load_predictor.predictor, data_load)
        self._retrain_and_log("PV", self.pv_predictor.predictor, data_pv)
        self._retrain_and_log("Price", self.price_predictor.predictor, data_price)

    @staticmethod
    def _retrain_and_log(name:str, predictor:BasePredictor, data:PredictionModelSets) -> None:
        train_loss, _ = predictor.fit(data)
        print(f"{name} predictor retrained with {data.X_train.shape[0]} samples, loss:  {train_loss}.")

    def _prepare_training_data(
            self,
            data_stack: np.ndarray,
            y_pos:int,
            scaler: Optional[StandardScaler],
            two_d: bool
    ) -> PredictionModelSets:
        if scaler: data_stack = scaler.transform(data_stack)
        X = np.zeros((self.retrain_interval, self.history_length, data_stack.shape[1]), dtype=np.float32)
        y = np.zeros((self.retrain_interval, self.prediction_horizon), dtype=np.float32)
        for i in range(self.retrain_interval):
            history_end = i + self.history_length
            X[i] = data_stack[i: history_end, :]
            y[i] = data_stack[history_end:history_end + self.prediction_horizon, y_pos]
        if two_d: X = X.reshape(X.shape[0], -1)
        return PredictionModelSets(X_train=X, y_train=y, target_indexes=np.array([y_pos]))

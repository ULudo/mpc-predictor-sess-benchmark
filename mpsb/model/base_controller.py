from abc import abstractmethod
import csv
from pathlib import Path
from typing import Dict, List, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from mpsb.env import BuildingEnv
from mpsb.util.consts_and_types import InfoMsg

@dataclass
class MonitoringMsg:
    episode_len: int = 0
    episode_return: float = 0.0
    env_metrics: Dict[str, float] = None
    trajectory: Dict[str, list] = None
    predictions: Dict[str, list] = None

class BaseController:

    @abstractmethod
    def step(self, obs:np.ndarray, info:InfoMsg) -> int | float | np.ndarray:
        """
        Compute the action to be taken by the controller based on the current observation and additional information.

        Args:
            obs (np.ndarray): The current observation from the environment.
            info (InfoMsg): Additional information provided by the environment.

        Returns:
            int | float | np.ndarray: The action to be taken by the controller.
        """
        return 0.0

def run_eval_loop(
        env:BuildingEnv,
        controller:BaseController,
        heat_up_steps:int = 0
) -> MonitoringMsg:

    # Variables for monitoring
    monitoring = defaultdict(list)
    # rows: [time, true, pred_0, ..., pred_H-1]
    pred_rows = {"load": [], "pv": [], "price": []}
    first_ts = 0
    episode_return = 0
    episode_price = 0
    episode_consumption = 0
    episode_len = 0

    # Reset the environment
    steps = 0
    done = False
    obs, info = env.reset()
    while not done:

        # Execute environment transaction
        action = controller.step(obs, info)

        # log predictions if available
        if hasattr(controller, "_last_forecasts") and getattr(controller, "_last_forecasts") is not None:
            ts = info.time
            truth = getattr(controller, "_last_truth", None)
            forecasts = controller._last_forecasts
            if truth is not None:
                def _mk_row(var_name: str):
                    arr = forecasts[var_name]
                    preds = arr[1:].tolist() if hasattr(arr, "ndim") and arr.ndim > 0 else [float(arr)]
                    return [ts, float(truth[var_name])] + preds
                pred_rows["load"].append(_mk_row("load"))
                pred_rows["pv"].append(_mk_row("pv"))
                pred_rows["price"].append(_mk_row("price"))

        obs, rew, terminal, terminated, info = env.step(action)
        done = terminal or terminated

        # Monitoring
        if steps >= heat_up_steps:
            # TODO: Not considering array-like actions
            if steps == heat_up_steps:
                first_ts = info.time
            monitoring["actions"].append(action)
            monitoring["rewards"].append(rew)
            state = info.state
            for k, v in asdict(state).items():
                monitoring[k].append(v)
            episode_return += rew
            # TODO: Env dependent variables (not generic)
            episode_price += state.price
            episode_consumption += state.e_bat + state.load - state.pv_gen
            episode_len += 1
        steps += 1

    return MonitoringMsg(
        episode_len=episode_len,
        episode_return=episode_return,
        env_metrics={
            "first_ts": first_ts,
            "total_price": episode_price,
            "total_consumption": episode_consumption
        },
        trajectory=monitoring,
        predictions=pred_rows
    )


def evaluate_and_report(
        log_dir: Path,
        n_runs: int,
        envs:List[BuildingEnv],
        controller:BaseController,
        heat_up_steps:int = 0
) -> None:
    eval_metrics = defaultdict(list)
    eval_csv, trajectory_dir, predictions_dir = _create_dir_structure(log_dir)

    for run in range(n_runs):
        for env_idx, env in enumerate(envs):
            env_name = f"eval_run_{run}_of_env_{env_idx}"
            msg = run_eval_loop(env, controller, heat_up_steps)
            _write_evaluation_results(env_idx, env_name, eval_csv, msg, run)
            _create_trajectory_csv(env_name, msg, trajectory_dir)
            _write_prediction_csvs(env_name, msg.predictions, predictions_dir)
            eval_metrics["episode_len"].append(msg.episode_len)
            eval_metrics["episode_return"].append(msg.episode_return)
            for k, v in msg.env_metrics.items():
                eval_metrics[k].append(v)

    for k, v in eval_metrics.items():
        print(f"{k}: {np.mean(v)} +- {np.std(v)}")


def _create_trajectory_csv(
        env_name:str,
        msg:MonitoringMsg,
        trajectory_dir:Path
) -> None:
    fieldnames = msg.trajectory.keys()
    trajectory_csv = trajectory_dir / f"{env_name}.csv"
    with open(trajectory_csv, 'a', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(zip(*[msg.trajectory[key] for key in fieldnames]))


def _write_evaluation_results(
        env_idx:int,
        env_name:str,
        eval_csv:Path,
        msg:MonitoringMsg,
        run:int
) -> None:
    result = {
        "name": env_name,
        "return": msg.episode_return,
        "env_steps": msg.episode_len}
    for k, v in msg.env_metrics.items():
        result[k] = v
    result["run"] = run
    result["env_idx"] = env_idx
    fieldnames = result.keys()
    with open(eval_csv, 'a', encoding='UTF8', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerow(result)


def _create_dir_structure(log_dir:Path) -> Union[Path, Path, Path]:
    eval_dir = log_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    eval_csv = eval_dir / "evaluation_results.csv"
    trajectories_dir = eval_dir / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)
    predictions_dir = eval_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    return eval_csv, trajectories_dir, predictions_dir

def _write_prediction_csvs(env_name: str, pred_dict: Dict[str, list], pred_dir: Path) -> None:
    for var, rows in pred_dict.items():
        if not rows:
            continue
        csv_path = pred_dir / f"{env_name}_{var}.csv"
        with open(csv_path, 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                header = ["time", "true"] + [f"pred_{i}" for i in range(len(rows[0]) - 2)]
                writer.writerow(header)
            writer.writerows(rows)
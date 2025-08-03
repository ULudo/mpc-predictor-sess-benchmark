import shutil
from pathlib import Path
from time import gmtime, strftime
from typing import Union, List, Tuple

from omegaconf import OmegaConf, DictConfig, ListConfig

from mpsb.env import BuildingEnv, BuildingDataManager
from mpsb.model.base_controller import evaluate_and_report, BaseController
from mpsb.model.mpc import MPCOptimizer, PerfectMPController, PredictorBundle, PredictorMPController, \
    RetrainPredictorMPController
from mpsb.predictor_manager import PredictorManager
from mpsb.util.consts_and_types import ScalingType, FormatType, ActionSpaceType, ModelType, DataColumn
from mpsb.util.functions import check_path


class ConfigManager:

    def __init__(self, config_path: str):
        self.config, self.config_path = self._read_config(config_path)

    @staticmethod
    def _read_config(config_path: str) -> Tuple[DictConfig, Path]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return OmegaConf.load(config_path), config_path

    def run(self):
        log_dir = self._create_log_dir()
        self._copy_config_to_log_dir(log_dir)
        self._init_data_mgr(self.config.data)
        model_type = ModelType[self.config.model.name.upper()]
        if model_type == ModelType.MPC:
            self._run_mpc(log_dir)
        elif model_type == ModelType.PREDICTOR:
            two_d = self.config.model.get("two_d", False)
            mgr = PredictorManager(self.config, log_dir)
            mgr.prepare_data(two_d=two_d)
            mgr.save_scaler()
            if self.config.model.action == "train":
                mgr.train()
            elif self.config.model.action == "evaluate":
                # TODO: Load scaler (optional)
                predictor = mgr.load_predictor()
                mgr.evaluate(predictor)
            elif self.config.model.action == "tune":
                mgr.tune()
            elif self.config.model.action == "train_and_evaluate":
                mgr.train_and_evaluate()
            else:
                raise ValueError(f"Action {self.config.model.action} not supported")
        elif model_type == ModelType.DRL:
            raise NotImplementedError("DRL model not implemented yet")
        elif model_type == ModelType.BASE:
            self._run_base(log_dir)
        else:
            raise ValueError(f"Model {self.config.model.name} not supported")


    def _copy_config_to_log_dir(self, log_dir: Path) -> None:
        config_copy_path = log_dir / self.config_path.name
        shutil.copy(self.config_path, config_copy_path)

    def _run_base(self, log_dir: Path):
        evaluate_and_report(
            log_dir=log_dir,
            n_runs=self.config.get("n_eval_runs", 1),
            envs=self._create_envs(self.config.environment),
            controller=BaseController(),
            heat_up_steps=self.config.model.get('history_length', 0)
        )

    def _run_mpc(self, log_dir: Path):
        self._check_scaling_for_mpc()
        self._check_action_space_type_for_mpc()
        mpc_optimizer = MPCOptimizer(
            n_predictions=self.config.model.prediction_length,
            bat_efficiency=self.config.environment.battery.efficiency,
            bat_capacity=self.config.environment.battery.capacity,
            bat_max_power=self.config.environment.battery.max_power,
            tax=self.config.environment.get("tax", 0.0)
        )

        if self.config.model.mpc_type == "perfect":
            controller = PerfectMPController(mpc_optimizer)
        elif self.config.model.mpc_type in ["predictor", "retrain_predictor"]:
            predictors = {
                f"{name}_predictor": self._load_predictor(name, self.config.model[name])
                for name in  [DataColumn.LOAD, DataColumn.PV, DataColumn.PRICE]
            }
            if self.config.model.mpc_type == "predictor":
                controller = PredictorMPController(
                    **predictors,
                    optimizer=mpc_optimizer,
                    history_length=self.config.model.history_length,
                )
            elif self.config.model.mpc_type == "retrain_predictor":
                controller = RetrainPredictorMPController(
                    **predictors,
                    optimizer=mpc_optimizer,
                    history_length=self.config.model.history_length,
                    prediction_horizon=self.config.model.prediction_length,
                    retrain_interval=self.config.model.retrain_interval,
                )
        else:
            raise ValueError(f"Unknown MPC type: {self.config.model.mpc_type}")

        evaluate_and_report(
            log_dir=log_dir,
            n_runs=self.config.get("n_eval_runs", 1),
            envs=self._create_envs(self.config.environment),
            controller=controller,
            heat_up_steps=self.config.model.get('history_length', 0)
        )

    @staticmethod
    def _load_predictor(target:str, predictor_config: DictConfig) -> PredictorBundle:
        predictor_class = PredictorManager.get_predictor(predictor_config.predictor_type)
        predictor = predictor_class(**predictor_config.get("kwargs", {}))
        predictor.load(predictor_config.predictor_path)
        scaler_target_idx = 1 if target == DataColumn.PV else 0
        if scaler_path := getattr(predictor_config, "scaler_path", None):
            scaler = PredictorManager.load_scaler(scaler_path)
        else:
            scaler = None
        return PredictorBundle(
                predictor=predictor,
                scaler=scaler,
                scaler_target_idx=scaler_target_idx,
                two_d=predictor_config.two_d,
            )

    @staticmethod
    def _init_data_mgr(conf: DictConfig):
        datasets = conf.datasets
        if not datasets or not isinstance(datasets, ListConfig):
            raise ValueError("No datasets provided")
        house_data = {
            getattr(ds, 'name', f"house_{idx}"): check_path(ds.house_data_file)
            for idx, ds in enumerate(datasets)
        }
        price_data = check_path(conf.price_data_file)
        data_format = FormatType[conf.data_format.upper()]
        BuildingDataManager.load_datasets(
            datasets=house_data,
            price_data_file=price_data,
            input_format=data_format,
        )

    def _create_log_dir(self) -> Path:
        get_time_string = lambda: strftime("%Y%m%d_%H%M%S", gmtime())
        out_dir = getattr(self.config, "out_dir", "./out")
        name = getattr(self.config, "name", "experiment")
        log_dir = Path(out_dir) / name / get_time_string()
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    @staticmethod
    def _create_building_env_from_config(env_config: DictConfig) -> BuildingEnv:
        if dataset_args := env_config.get("dataset_args", None):
            dataset_args = OmegaConf.to_container(dataset_args, resolve=True)
        return BuildingEnv(
            dataset_args=dataset_args,
            battery_efficiency=env_config.get("battery_efficiency", 0.95),
            battery_max_power=env_config.get("battery_max_power", 8000),
            battery_capacity=env_config.get("battery_capacity", 20000),
            use_time_features=env_config.get("use_time_features", True),
            random_time_init=env_config.get("random_time_init", False),
            random_soc_init=env_config.get("random_soc_init", False),
            init_soc=env_config.get("init_soc", 0.5),
            init_time=env_config.get("init_time", None),
            episode_length=env_config.get("episode_length", None),
            tax=env_config.get("tax", 0.0),
            apply_deadband=env_config.get("apply_deadband", False),
            scaling_method=env_config.get("scaling_method", 'none'),
            load_stats=env_config.get("load_stats", None),
            price_stats=env_config.get("price_stats", None),
            pv_stats=env_config.get("pv_stats", None),
            prediction_horizon=env_config.get("prediction_horizon", 0),
            action_space_type=env_config.get("action_space_type", 'discrete')
        )

    @staticmethod
    def _create_envs(env_config: DictConfig) -> Union[List[BuildingEnv]]:
        num_envs = getattr(env_config, "num_envs", 1)

        def make_env(conf):
            return lambda: ConfigManager._create_building_env_from_config(conf)

        env_fun_list = [make_env(env_config) for _ in range(num_envs)]
        return [env() for env in env_fun_list]

    def _check_action_space_type_for_mpc(self):
        action_space_type = self.config.environment.action_space_type.upper()
        action_space_type = ActionSpaceType[action_space_type]
        assert action_space_type is ActionSpaceType.CONTINUOUS, \
            "Evaluation env must have continuous action space for MPC"

    def _check_scaling_for_mpc(self):
        scaling_method = self.config.environment.scaling_method.upper()
        scaling_method = ScalingType[scaling_method]
        assert scaling_method is ScalingType.NONE, \
            "Environment observations should not be scaled for MPC"

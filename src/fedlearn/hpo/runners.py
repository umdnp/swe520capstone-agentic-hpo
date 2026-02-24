from __future__ import annotations

import logging
from typing import Protocol

import optuna
from flwr.app import Context, ArrayRecord
from flwr.common import ConfigRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result, Strategy
from sklearn.pipeline import Pipeline

from fedlearn.common.config import HParams, ServerSettings, get_server_settings
from fedlearn.common.model import get_model, get_model_params, set_initial_params

logger = logging.getLogger(__name__)

# Constants

OPTUNA_SEED = 42


def _run_fl(
        *,
        strategy: Strategy,
        grid: Grid,
        hp: HParams,
        settings: ServerSettings,
        train_cfg: ConfigRecord | None = None,
        eval_cfg: ConfigRecord | None = None,
) -> tuple[Result, Pipeline]:
    """
    Run one FL execution with freshly initialized model parameters.
    """
    model = get_model(hp)
    set_initial_params(model)
    arrays = ArrayRecord(get_model_params(model))

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=settings.num_rounds,
    )
    return result, model


class ExperimentRunner(Protocol):
    def run(self, grid: Grid, context: Context) -> tuple[Result, Pipeline]:
        ...


class BaselineRunner:
    def run(self, grid: Grid, context: Context) -> tuple[Result, Pipeline]:
        settings = get_server_settings(context)
        base_hp = HParams.from_run_config(context)

        strategy = FedAvg(
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        # baseline clients use run_config (no config record sent)
        return _run_fl(strategy=strategy, grid=grid, hp=base_hp, settings=settings)


class StaticHPORunner:
    @staticmethod
    def _score_last_round(result: Result, metric: str) -> float:
        eval_metrics = getattr(result, "evaluate_metrics_clientapp", None)
        if not eval_metrics:
            raise RuntimeError(
                "No client evaluate metrics found in Result. "
                "Confirm your @app.evaluate() returns MetricRecord and server aggregates it."
            )

        last_round = max(eval_metrics.keys())
        rec = eval_metrics[last_round]
        if metric not in rec:
            raise KeyError(f"Metric {metric!r} not found. Available: {list(rec.keys())}")

        return float(rec[metric])

    @staticmethod
    def _suggest_hparams(trial: optuna.trial.BaseTrial, base: HParams) -> HParams:
        local_epochs = trial.suggest_int("local-epochs", 1, 10)
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        lr_sched = trial.suggest_categorical(
            "sgd-learning-rate", ["optimal", "constant", "adaptive"]
        )

        eta0 = (
            float(trial.suggest_float("sgd-eta0", 1e-4, 1e-1, log=True))
            if lr_sched in ("constant", "adaptive")
            else 0.0
        )

        return HParams(
            local_epochs=local_epochs,
            penalty=penalty,
            class_weight_cfg=base.class_weight_cfg,
            sgd_learning_rate=lr_sched,
            sgd_eta0_cfg=eta0,
        )

    def run(self, grid: Grid, context: Context) -> tuple[Result, Pipeline]:
        settings = get_server_settings(context)
        base_hp = HParams.from_run_config(context)

        n_trials = int(context.run_config.get("hpo-n-trials", 15))
        trial_rounds = int(context.run_config.get("hpo-num-rounds", 2))
        metric = str(context.run_config.get("hpo-metric", "roc_auc"))
        direction = str(context.run_config.get("hpo-direction", "maximize"))

        # shorter settings for each trial
        trial_settings = ServerSettings(
            num_rounds=trial_rounds,
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        def objective(trial: optuna.Trial) -> float:
            hp_trial = self._suggest_hparams(trial, base_hp)
            cfg_trial = hp_trial.to_config()

            trial_strategy = FedAvg(
                fraction_train=settings.fraction_train,
                fraction_evaluate=settings.fraction_evaluate,
            )

            result, _ = _run_fl(
                strategy=trial_strategy,
                grid=grid,
                hp=hp_trial,
                settings=trial_settings,
                train_cfg=cfg_trial,
                eval_cfg=cfg_trial,
            )
            return self._score_last_round(result, metric=metric)

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        )
        study.optimize(objective, n_trials=n_trials)

        # rebuild the best_hp from best_params
        best_hp = self._suggest_hparams(optuna.trial.FixedTrial(study.best_params), base_hp)

        logger.info(f"[static_hpo] best_value={study.best_value}, best_params={study.best_params}")

        best_cfg = best_hp.to_config()

        final_strategy = FedAvg(
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        # full run using best static config
        return _run_fl(
            strategy=final_strategy,
            grid=grid,
            hp=best_hp,
            settings=settings,
            train_cfg=best_cfg,
            eval_cfg=best_cfg,
        )


class AgenticFedAvg(FedAvg):
    """
    TODO: simple impl for now
    """

    def __init__(self, *, seed_hp: HParams, **kwargs):
        super().__init__(**kwargs)
        self.seed_hp = seed_hp

    def configure_train(self, server_round, arrays, config, grid):
        hp = self.seed_hp

        # sample heuristic schedule; swap this with an agent/controller later
        if server_round == 1:
            hp = HParams(
                local_epochs=1,
                penalty=hp.penalty,
                class_weight_cfg=hp.class_weight_cfg,
                sgd_learning_rate=hp.sgd_learning_rate,
                sgd_eta0_cfg=hp.sgd_eta0_cfg,
            )
        elif server_round == 2:
            hp = HParams(
                local_epochs=2,
                penalty=hp.penalty,
                class_weight_cfg=hp.class_weight_cfg,
                sgd_learning_rate=hp.sgd_learning_rate,
                sgd_eta0_cfg=hp.sgd_eta0_cfg,
            )

        return super().configure_train(server_round, arrays, hp.to_config(), grid)


class AgenticHPORunner:
    def run(self, grid: Grid, context: Context) -> tuple[Result, Pipeline]:
        settings = get_server_settings(context)
        seed_hp = HParams.from_run_config(context)
        seed_cfg = seed_hp.to_config()

        strategy = AgenticFedAvg(
            seed_hp=seed_hp,
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        return _run_fl(
            strategy=strategy,
            grid=grid,
            hp=seed_hp,
            settings=settings,
            train_cfg=seed_cfg,  # initial seed sent
            eval_cfg=seed_cfg,
        )

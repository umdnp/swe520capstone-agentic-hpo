from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol

import optuna
from flwr.app import ArrayRecord, Context
from flwr.common import ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result, Strategy
from sklearn.pipeline import Pipeline

from fedlearn.common.config import DataSplit, HParams, ServerSettings, get_server_settings
from fedlearn.common.config import HP_LOCAL_EPOCHS, HP_PENALTY, HP_LR_SCHEDULE, HP_ETA0
from fedlearn.common.model import get_model, get_model_params, set_initial_params
from fedlearn.hpo.agents import AgenticFedAvg, AgenticHPOController

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


def _evaluate_final_arrays(
        *,
        grid: Grid,
        arrays: ArrayRecord,
        settings: ServerSettings,
        eval_cfg: ConfigRecord,
) -> Result:
    """
    Run a final evaluation-only pass using already-trained global arrays.
    """
    strategy = EvalOnlyFedAvg(
        fraction_train=settings.fraction_train,
        fraction_evaluate=settings.fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(),
        evaluate_config=eval_cfg,
        num_rounds=1,
    )
    return result


class ExperimentRunner(Protocol):
    """
    Base class for all experiment runners.
    """

    def run(self, grid: Grid, context: Context) -> tuple[Result, Pipeline]:
        ...


class BaselineRunner:
    """
    Baseline federated training with fixed hyperparameters.
    """

    def run(self, grid: Grid, context: Context) -> tuple[Result, Pipeline]:
        settings = get_server_settings(context)
        base_hp = HParams.from_run_config(context)

        strategy = FedAvg(
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        baseline_cfg = base_hp.to_config(
            train_split=DataSplit.TRAIN_VAL,
            eval_split=DataSplit.TEST,
        )

        return _run_fl(
            strategy=strategy,
            grid=grid,
            hp=base_hp,
            settings=settings,
            train_cfg=baseline_cfg,
            eval_cfg=baseline_cfg,
        )


class StaticHPORunner:
    """
    Federated training using static hyperparameters (Optuna tuned).
    """

    @staticmethod
    def _score_static_trial(
            result: Result,
            auc_metric: str = "roc_auc",
            loss_metric: str = "loss",
            last_k: int = 3,
            loss_penalty_weight: float = 0.02,
    ) -> float:
        eval_metrics = getattr(result, "evaluate_metrics_clientapp", None)
        if not eval_metrics:
            raise RuntimeError("No client evaluate metrics found in Result.")

        rounds = sorted(eval_metrics.keys())
        tail = rounds[-last_k:]

        aucs = [float(eval_metrics[r][auc_metric]) for r in tail]
        losses = [float(eval_metrics[r][loss_metric]) for r in tail]

        mean_auc = sum(aucs) / len(aucs)
        mean_loss = sum(losses) / len(losses)

        return mean_auc - loss_penalty_weight * mean_loss

    @staticmethod
    def _suggest_hparams(trial: optuna.trial.BaseTrial, base: HParams) -> HParams:
        local_epochs = trial.suggest_int(HP_LOCAL_EPOCHS, 3, 8)
        penalty = trial.suggest_categorical(HP_PENALTY, ["l2", "l1", "elasticnet"])
        lr_sched = trial.suggest_categorical(
            HP_LR_SCHEDULE, ["optimal", "constant", "adaptive"]
        )

        eta0 = (
            float(trial.suggest_float(HP_ETA0, 1e-4, 1e-2, log=True))
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
        """
        Run Optuna-based static HPO with:
        - trial-time training on TRAIN
        - trial-time evaluation on VALIDATION
        - final training on TRAIN_VAL
        - final evaluation on TEST
        """
        settings = get_server_settings(context)
        base_hp = HParams.from_run_config(context)

        n_trials = int(context.run_config.get("hpo-n-trials", 15))
        trial_rounds = int(context.run_config.get("hpo-num-rounds", 5))
        direction = str(context.run_config.get("hpo-direction", "maximize"))

        # shorter settings for each trial
        trial_settings = ServerSettings(
            num_rounds=trial_rounds,
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        def objective(trial: optuna.Trial) -> float:
            hp_trial = self._suggest_hparams(trial, base_hp)

            cfg_trial = hp_trial.to_config(
                train_split=DataSplit.TRAIN,
                eval_split=DataSplit.VALIDATION,
            )

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

            return self._score_static_trial(result)

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        )
        study.optimize(objective, n_trials=n_trials)

        # rebuild the best_hp from best_params
        best_hp = self._suggest_hparams(
            optuna.trial.FixedTrial(study.best_params),
            base_hp,
        )

        logger.info("[static_hpo] best_value=%s, best_params=%s", study.best_value, study.best_params)

        best_cfg = best_hp.to_config(
            train_split=DataSplit.TRAIN_VAL,
            eval_split=DataSplit.TEST,
        )

        final_strategy = FedAvg(
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        # full run using the best static config on the final phase
        return _run_fl(
            strategy=final_strategy,
            grid=grid,
            hp=best_hp,
            settings=settings,
            train_cfg=best_cfg,
            eval_cfg=best_cfg,
        )


class EvalOnlyFedAvg(FedAvg):
    """
    Strategy that skips training and performs exactly one federated evaluation round.
    """

    def configure_train(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid,
    ) -> Iterable[Message]:
        return []

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        return None, None


class AgenticHPORunner:
    """
    Federated training with agent-controlled hyperparameters.

    Training rounds use VALIDATION for control. After training completes, a final one-shot
    TEST evaluation is run on the learned global model without additional training.
    """

    def run(self, grid: Grid, context: Context) -> tuple[Result, Pipeline]:
        settings = get_server_settings(context)
        seed_hp = HParams.from_run_config(context)
        seed_cfg = seed_hp.to_config(
            train_split=DataSplit.TRAIN,
            eval_split=DataSplit.VALIDATION,
        )

        rc = context.run_config
        model = str(rc.get("agent-model", "gpt-5.2"))
        temperature = float(rc.get("agent-temperature", 0.2))
        total_rounds = int(rc.get("num-server-rounds", 20))

        controller = AgenticHPOController(
            model=model,
            temperature=temperature,
            total_rounds=total_rounds,
        )

        strategy = AgenticFedAvg(
            seed_hp=seed_hp,
            controller=controller,
            fraction_train=settings.fraction_train,
            fraction_evaluate=settings.fraction_evaluate,
        )

        train_result, model_pipeline = _run_fl(
            strategy=strategy,
            grid=grid,
            hp=seed_hp,
            settings=settings,
            train_cfg=seed_cfg,
            eval_cfg=seed_cfg,
        )

        if train_result.arrays is None:
            raise RuntimeError("Agentic run completed without final arrays.")

        # one-shot final TEST evaluation using the learned global model
        final_test_cfg = seed_hp.to_config(
            train_split=DataSplit.TRAIN,
            eval_split=DataSplit.TEST,
        )

        test_result = _evaluate_final_arrays(
            grid=grid,
            arrays=train_result.arrays,
            settings=settings,
            eval_cfg=final_test_cfg,
        )

        # keep the trained global arrays from the agentic run
        test_result.arrays = train_result.arrays

        return test_result, model_pipeline

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from agents import Agent, ModelSettings, Runner
from flwr.app import ArrayRecord, ConfigRecord
from flwr.common.message import Message
from flwr.common.record.metricrecord import MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from openai import OpenAIError
from pydantic import BaseModel, Field, ValidationError, model_validator

from fedlearn.common.config import DataSplit, HParams
from fedlearn.common.metrics import metricrecord_to_dict

logger = logging.getLogger(__name__)

# Constants

Penalty = Literal["l1", "l2", "elasticnet"]
Schedule = Literal["optimal", "constant", "adaptive"]

ALLOWED_PENALTIES = list(get_args(Penalty))
ALLOWED_SCHEDULES = list(get_args(Schedule))


def _safe_float(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) else None


def _mean_last(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def _delta_window(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return values[-1] - values[-n]


class AgenticHPOProposal(BaseModel):
    """
    Structured agent output for next-round federated hyperparameters.
    """
    local_epochs: int = Field(ge=3, le=8)
    penalty: Penalty
    sgd_learning_rate: Schedule
    sgd_eta0: float = Field(ge=0.0, le=1e-2)
    exploit: Literal[0, 1]

    @model_validator(mode="after")
    def validate_eta0(self) -> "AgenticHPOProposal":
        if self.sgd_learning_rate == "optimal":
            self.sgd_eta0 = 0.0
        elif not (1e-4 <= self.sgd_eta0 <= 1e-2):
            raise ValueError("sgd_eta0 must be in [1e-4, 1e-2] for constant/adaptive.")
        return self


@dataclass(slots=True)
class AgenticHPOController:
    """
    LLM-based controller that proposes next-round HParams from aggregated history.
    """
    model: str = "gpt-5.2"
    temperature: float = 0.2
    total_rounds: int = 20
    max_history_rounds: int = 12

    _enabled: bool = field(init=False)
    _agent: Agent | None = field(init=False, default=None)
    _exploit_by_round: dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # if no key configured, allow FL to run (seed-only behavior)
        self._enabled = bool(os.environ.get("OPENAI_API_KEY", "").strip())

        if not self._enabled:
            logger.warning("Agent disabled (OPENAI_API_KEY missing); using base_hp only")
            return

        logger.info("Agent enabled (OPENAI_API_KEY found)")

        self._agent = Agent(
            name="Federated HPO controller",
            instructions=(
                "You are an expert federated learning hyperparameter controller. "
                "Each round, propose the next training hyperparameters. "
                "You only see aggregated metrics and prior choices. "
                "Goal: maximize roc_auc while keeping loss low and training stable. "
                "Do not overreact to one noisy round. "
                "Early rounds may explore more. Later rounds should prefer smaller, conservative changes. "
                "Treat changes to penalty and learning-rate schedule as major changes. "
                "Prefer adjusting local_epochs or eta0 before changing penalty or schedule. "
                "Use constant learning rate only when there is clear evidence that the current learning-rate approach is underperforming. "
                "Avoid changing multiple major hyperparameters at once unless performance has clearly worsened for multiple rounds. "
                "Set exploit=1 only when you are intentionally keeping or only slightly adjusting a configuration that appears to be working. "
                "Set exploit=0 when you are testing a meaningfully different configuration. "
            ),
            model=self.model,
            model_settings=ModelSettings(temperature=self.temperature),
            output_type=AgenticHPOProposal,
        )

    def get_exploit(self, server_round: int) -> int | None:
        return self._exploit_by_round.get(int(server_round))

    @staticmethod
    def _build_history_summary(recent: list[dict[str, Any]]) -> dict[str, Any]:
        aucs: list[float] = []
        losses: list[float] = []

        for rec in recent:
            metrics = rec.get("metrics", {})
            auc = _safe_float(metrics.get("roc_auc"))
            loss = _safe_float(metrics.get("loss"))

            if auc is not None:
                aucs.append(auc)
            if loss is not None:
                losses.append(loss)

        auc_delta_5 = _delta_window(aucs, 5)
        loss_delta_5 = _delta_window(losses, 5)

        return {
            "auc_last": aucs[-1] if aucs else None,
            "loss_last": losses[-1] if losses else None,
            "auc_mean_3": _mean_last(aucs, 3),
            "auc_mean_5": _mean_last(aucs, 5),
            "loss_mean_3": _mean_last(losses, 3),
            "loss_mean_5": _mean_last(losses, 5),
            "auc_delta_5": auc_delta_5,
            "loss_delta_5": loss_delta_5,
            "plateau": auc_delta_5 is not None and abs(auc_delta_5) < 0.002,
        }

    def propose_next(
            self,
            *,
            base_hp: HParams,
            server_round: int,
            history: list[dict[str, Any]],
    ) -> HParams:
        """
        Return next-round HParams, falling back to base_hp on any failure.
        """
        if not self._enabled or self._agent is None:
            return base_hp

        recent = history[-self.max_history_rounds:]
        summary = self._build_history_summary(recent)
        explore_phase = server_round <= math.ceil(0.25 * self.total_rounds)

        rules = (
            [
                "Exploration phase: moderate experimentation is allowed.",
                "Prefer changing one dimension at a time.",
                "Prefer adjusting local_epochs or eta0 before changing penalty or learning-rate schedule.",
                "Treat penalty and learning-rate schedule changes as major changes.",
                "Use constant learning rate only when there is clear evidence that the current schedule is underperforming.",
                "If metrics improve strongly, keep similar parameters next round.",
                "Use exploit=0 when testing a meaningfully different configuration.",
            ]
            if explore_phase
            else [
                "Stabilization phase: prefer keeping the current parameters unless there is strong evidence to change.",
                "Use history_summary more than any single round.",
                "If roc_auc improves and loss does not worsen across multiple rounds, keep similar parameters.",
                "If metrics stall or worsen for multiple rounds, change only one dimension at a time.",
                "Prefer adjusting local_epochs or eta0 before changing penalty or learning-rate schedule.",
                "Treat penalty and learning-rate schedule changes as major changes.",
                "Use constant learning rate only when there is clear evidence that the current schedule is underperforming.",
                "If plateau is true, keep parameters or make only a very small change.",
                "Use exploit=1 only when keeping or slightly adjusting a configuration that appears to be working.",
            ]
        )

        payload = {
            "objective": {"maximize": "roc_auc", "minimize": "loss"},
            "round": int(server_round),
            "phase": "exploration" if explore_phase else "stabilization",
            "search_space": {
                "local_epochs": [3, 8],
                "penalty": ALLOWED_PENALTIES,
                "sgd_learning_rate": ALLOWED_SCHEDULES,
                "sgd_eta0": "if constant/adaptive: [1e-4, 1e-2]; if optimal: 0.0",
                "exploit": [0, 1],
            },
            "current_hp": {
                "local_epochs": base_hp.local_epochs,
                "penalty": base_hp.penalty,
                "class_weight_cfg": base_hp.class_weight_cfg,
                "sgd_learning_rate": base_hp.sgd_learning_rate,
                "sgd_eta0_cfg": base_hp.sgd_eta0_cfg,
            },
            "history": recent,
            "history_summary": summary,
            "rules": rules,
        }

        prompt = (
            "Choose next-round federated training hyperparameters.\n"
            "Return only the structured output.\n\n"
            f"STATE JSON:\n{json.dumps(payload, indent=2)}"
        )

        try:
            result = Runner.run_sync(self._agent, prompt)
            proposal = result.final_output

            if not isinstance(proposal, AgenticHPOProposal):
                raise TypeError(f"Unexpected output type: {type(proposal)}")

            self._exploit_by_round[int(server_round)] = int(proposal.exploit)

        except (ValidationError, ValueError, TypeError) as e:
            logger.warning("Agent proposal invalid; falling back. err=%s", e)
            return base_hp
        except (OpenAIError, RuntimeError):
            logger.exception("Agent call failed; falling back to base_hp.")
            return base_hp

        return HParams(
            local_epochs=proposal.local_epochs,
            penalty=proposal.penalty,
            class_weight_cfg=base_hp.class_weight_cfg,
            sgd_learning_rate=proposal.sgd_learning_rate,
            sgd_eta0_cfg=proposal.sgd_eta0,
        )


class AgenticFedAvg(FedAvg):
    """
    FedAvg strategy that uses an LLM controller to choose HParams each round.
    """

    def __init__(self, *, seed_hp: HParams, controller: AgenticHPOController, **kwargs):
        super().__init__(**kwargs)
        self.seed_hp = seed_hp
        self.controller = controller
        self._hp_by_round: dict[int, HParams] = {}
        self._history: list[dict[str, Any]] = []

    def _base_hp_for_round(self, server_round: int) -> HParams:
        """
        Use previous round hp as the baseline to encourage stability.
        """
        if server_round <= 1:
            return self.seed_hp
        return self._hp_by_round.get(server_round - 1, self.seed_hp)

    def configure_train(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid,
    ) -> Iterable[Message]:
        """
        Choose next-round hyperparameters and update the training config.
        """
        rnd = int(server_round)

        base_hp = self._base_hp_for_round(rnd)
        hp = base_hp if rnd == 1 else self.controller.propose_next(
            base_hp=base_hp,
            server_round=rnd,
            history=self._history,
        )

        self._hp_by_round[rnd] = hp

        exploit = self.controller.get_exploit(rnd)
        last = self._history[-1]["metrics"] if self._history else {}
        prev_auc = last.get("roc_auc")
        prev_loss = last.get("loss")

        logger.info(
            "[agentic_hpo] round=%d exploit=%s prev_auc=%s prev_loss=%s hp={local_epochs=%d penalty=%s lr=%s eta0=%.6g}",
            rnd,
            str(exploit) if exploit is not None else "NA",
            f"{prev_auc:.6f}" if isinstance(prev_auc, (int, float)) else "NA",
            f"{prev_loss:.6f}" if isinstance(prev_loss, (int, float)) else "NA",
            hp.local_epochs,
            hp.penalty,
            hp.sgd_learning_rate,
            float(hp.sgd_eta0_cfg),
        )

        hp_cfg = hp.to_config(
            train_split=DataSplit.TRAIN,
            eval_split=DataSplit.VALIDATION,
        )

        # merge into existing config (don’t clobber other keys)
        for k, v in hp_cfg.items():
            config[k] = v

        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
            self,
            server_round: int,
            arrays: ArrayRecord,
            config: ConfigRecord,
            grid: Grid,
    ) -> Iterable[Message]:
        """
        Ensure evaluation uses the same per-round config as training.
        """
        rnd = int(server_round)
        hp = self._hp_by_round.get(rnd, self.seed_hp)

        hp_cfg = hp.to_config(
            train_split=DataSplit.TRAIN,
            eval_split=DataSplit.VALIDATION,
        )

        for k, v in hp_cfg.items():
            config[k] = v

        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> MetricRecord | None:
        """
        Aggregate evaluation replies and record aggregated metrics for agent history.
        """
        mrec = super().aggregate_evaluate(server_round, replies)
        if mrec is None:
            return None

        rnd = int(server_round)
        hp = self._hp_by_round.get(rnd, self.seed_hp)

        metrics_dict = metricrecord_to_dict(mrec)

        rec: dict[str, Any] = {
            "round": rnd,
            "hp": {
                "local_epochs": hp.local_epochs,
                "penalty": hp.penalty,
                "sgd_learning_rate": hp.sgd_learning_rate,
                "sgd_eta0_cfg": hp.sgd_eta0_cfg,
            },
            "metrics": {k: float(v) for k, v in metrics_dict.items() if isinstance(v, (int, float))},
        }

        self._history.append(rec)
        return mrec

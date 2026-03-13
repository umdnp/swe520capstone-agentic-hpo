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

from fedlearn.common.config import HParams
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
    local_epochs: int = Field(ge=1, le=10)
    penalty: Penalty
    sgd_learning_rate: Schedule
    # eta0 is conditionally constrained; allow 0.0 so "optimal" can force it to 0.0.
    sgd_eta0: float = Field(ge=0.0, le=1e-1)
    exploit: Literal[0, 1]

    @model_validator(mode="after")
    def validate_eta0(self) -> "AgenticHPOProposal":
        """Enforce eta0 dependency on learning-rate schedule."""
        if self.sgd_learning_rate == "optimal":
            self.sgd_eta0 = 0.0
        else:
            if not (1e-4 <= self.sgd_eta0 <= 1e-1):
                raise ValueError("sgd_eta0 must be in [1e-4, 1e-1] for constant/adaptive.")
        return self


@dataclass
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
        if self._enabled:
            logger.info("Agent enabled (OPENAI_API_KEY found)")
        else:
            logger.warning("Agent disabled (OPENAI_API_KEY missing); using base_hp only")
            return

        self._agent = Agent(
            name="Federated HPO controller",
            instructions=(
                "You are an expert federated learning hyperparameter controller. "
                "Each round, propose the next training hyperparameters. "
                "You only see aggregated metrics and the history of prior choices. "
                "Goal: maximize roc_auc while keeping loss low and training stable. "
                "Do not overreact to one noisy round. Prefer consistency across multiple rounds. "
                "Prefer controlled changes. Early in training, exploration is acceptable, but later in training prefer small step changes and avoid changing multiple hyperparameters at once unless performance has clearly worsened for several rounds. "
                "Obey the allowed ranges and categories exactly."
            ),
            model=self.model,
            model_settings=ModelSettings(temperature=self.temperature),
            output_type=AgenticHPOProposal,
        )

    def get_exploit(self, server_round: int) -> int | None:
        """
        Return exploit flag for a round if available.
        """
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

        auc_last = aucs[-1] if aucs else None
        loss_last = losses[-1] if losses else None

        auc_mean_3 = _mean_last(aucs, 3)
        auc_mean_5 = _mean_last(aucs, 5)
        loss_mean_3 = _mean_last(losses, 3)
        loss_mean_5 = _mean_last(losses, 5)

        auc_delta_3 = _delta_window(aucs, 3)
        auc_delta_5 = _delta_window(aucs, 5)
        loss_delta_3 = _delta_window(losses, 3)
        loss_delta_5 = _delta_window(losses, 5)

        plateau = auc_delta_5 is not None and abs(auc_delta_5) < 0.002

        return {
            "auc_last": auc_last,
            "loss_last": loss_last,
            "auc_mean_3": auc_mean_3,
            "auc_mean_5": auc_mean_5,
            "loss_mean_3": loss_mean_3,
            "loss_mean_5": loss_mean_5,
            "auc_delta_3": auc_delta_3,
            "auc_delta_5": auc_delta_5,
            "loss_delta_3": loss_delta_3,
            "loss_delta_5": loss_delta_5,
            "plateau": plateau,
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

        if explore_phase:
            rules = [
                "Do not ignore local_epochs during exploration.",
                "Early exploration of local training intensity is encouraged.",
                "Consider changing local_epochs as well as learning-rate settings.",
                "Explore local_epochs deliberately; prefer moderate values unless there is strong evidence to increase or decrease further.",
                "You may explore larger changes than later rounds, but still change only one major hyperparameter at a time.",
                "If metrics improve strongly, keep similar parameters for the next round.",
            ]
        else:
            rules = [
                "Prefer small changes unless metrics worsen or plateau.",
                "Use history_summary more than any single round when deciding.",
                "If roc_auc improves and loss decreases over multiple rounds, consider exploit=1 and keep similar params.",
                "If roc_auc stalls or degrades for 2+ rounds, explore by changing one dimension at a time.",
                "If plateau is true, prefer keeping the current parameters or making only a very small change.",
                "Do not change more than one major hyperparameter at once.",
                "Respect eta0 dependency and all category constraints.",
            ]

        payload = {
            "objective": {"maximize": "roc_auc", "minimize": "loss"},
            "round": int(server_round),
            "phase": "exploration" if explore_phase else "stabilization",
            "search_space": {
                "local_epochs": [1, 10],
                "penalty": ALLOWED_PENALTIES,
                "sgd_learning_rate": ALLOWED_SCHEDULES,
                "sgd_eta0": "if constant/adaptive: [1e-4, 1e-1]; if optimal: 0.0",
                "exploit": [0, 1],
            },
            "base_hp": {
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
        Use previous round hp as baseline to encourage stability.
        """
        if server_round <= 1:
            return self.seed_hp
        return self._hp_by_round.get(server_round - 1, self.seed_hp)

    def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """
        Choose next-round hyperparameters and update the config.
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

        hp_cfg = hp.to_config()

        # merge into existing config (don’t clobber other keys)
        for k, v in hp_cfg.items():
            config[k] = v

        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
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

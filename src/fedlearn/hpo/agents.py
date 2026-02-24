from __future__ import annotations

import json
import logging
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

ALLOWED_PENALTIES = tuple(get_args(Penalty))
ALLOWED_SCHEDULES = tuple(get_args(Schedule))

ALLOWED_PENALTIES_LIST = list(ALLOWED_PENALTIES)
ALLOWED_SCHEDULES_LIST = list(ALLOWED_SCHEDULES)


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
    max_history_rounds: int = 12

    _enabled: bool = field(init=False)
    _agent: Agent | None = field(init=False, default=None)
    _exploit_by_round: dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # If no key is configured, allow FL to run (seed-only behavior).
        self._enabled = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        if not self._enabled:
            return

        self._agent = Agent(
            name="Federated HPO controller",
            instructions=(
                "You are an expert federated learning hyperparameter controller. "
                "Each round, propose the next training hyperparameters. "
                "You only see aggregated metrics and the history of prior choices. "
                "Goal: maximize roc_auc while keeping loss low and training stable. "
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
    def _coerce_and_validate(p: AgenticHPOProposal) -> AgenticHPOProposal:
        """
        Defensive normalization and re-validation.
        """
        penalty = str(p.penalty).strip().lower()
        sched = str(p.sgd_learning_rate).strip().lower()

        if penalty not in ALLOWED_PENALTIES:
            penalty = "l2"
        if sched not in ALLOWED_SCHEDULES:
            sched = "optimal"

        return AgenticHPOProposal(
            local_epochs=min(max(int(p.local_epochs), 1), 10),
            penalty=penalty,  # type: ignore[arg-type]
            sgd_learning_rate=sched,  # type: ignore[arg-type]
            sgd_eta0=float(p.sgd_eta0),
            exploit=1 if int(p.exploit) == 1 else 0,
        )

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
        payload = {
            "objective": {"maximize": "roc_auc", "minimize": "loss"},
            "round": int(server_round),
            "search_space": {
                "local_epochs": [1, 10],
                "penalty": ALLOWED_PENALTIES_LIST,
                "sgd_learning_rate": ALLOWED_SCHEDULES_LIST,
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
            "rules": [
                "Prefer small changes unless metrics worsen or plateau.",
                "If roc_auc improves and loss decreases, consider exploit=1 and keep similar params.",
                "If roc_auc stalls or degrades for 2+ rounds, explore by changing one dimension at a time.",
                "Respect eta0 dependency and all category constraints.",
            ],
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

            proposal = self._coerce_and_validate(proposal)
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
        prev_metrics = self._history[-1] if self._history else None
        prev_auc = None
        prev_loss = None

        if prev_metrics:
            prev_auc = prev_metrics.get("metrics", {}).get("roc_auc")
            prev_loss = prev_metrics.get("loss")

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

        # capture loss key (if exists)
        if "loss" in metrics_dict and isinstance(metrics_dict["loss"], (int, float)):
            rec["loss"] = float(metrics_dict["loss"])

        self._history.append(rec)
        return mrec

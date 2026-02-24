from __future__ import annotations

from dataclasses import dataclass

from flwr.app import Context
from flwr.common import ConfigRecord, Message, RecordDict

HP_LOCAL_EPOCHS = "local-epochs"
HP_PENALTY = "penalty"
HP_CLASS_WEIGHT = "class-weight"
HP_LR_SCHEDULE = "sgd-learning-rate"
HP_ETA0 = "sgd-eta0"


@dataclass(frozen=True)
class HParams:
    local_epochs: int
    penalty: str
    class_weight_cfg: str
    sgd_learning_rate: str
    sgd_eta0_cfg: float

    @property
    def class_weight(self) -> str | None:
        return "balanced" if self.class_weight_cfg == "balanced" else None

    @property
    def sgd_eta0(self) -> float:
        sched = self.sgd_learning_rate
        return self.sgd_eta0_cfg if sched in ("constant", "adaptive") else 0.0

    def to_config(self) -> ConfigRecord:
        return ConfigRecord(
            {
                HP_LOCAL_EPOCHS: int(self.local_epochs),
                HP_PENALTY: str(self.penalty),
                HP_CLASS_WEIGHT: str(self.class_weight_cfg),
                HP_LR_SCHEDULE: str(self.sgd_learning_rate),
                HP_ETA0: float(self.sgd_eta0_cfg),
            }
        )

    @staticmethod
    def from_config(cfg: ConfigRecord) -> "HParams":
        return HParams(
            local_epochs=int(cfg.get(HP_LOCAL_EPOCHS)),
            penalty=str(cfg.get(HP_PENALTY)),
            class_weight_cfg=str(cfg.get(HP_CLASS_WEIGHT, "none")).strip().lower(),
            sgd_learning_rate=str(cfg.get(HP_LR_SCHEDULE, "optimal")).strip().lower(),
            sgd_eta0_cfg=float(cfg.get(HP_ETA0, 0.0)),
        )

    @staticmethod
    def from_run_config(context: Context) -> "HParams":
        rc = context.run_config
        return HParams(
            local_epochs=int(rc[HP_LOCAL_EPOCHS]),
            penalty=str(rc[HP_PENALTY]),
            class_weight_cfg=str(rc.get(HP_CLASS_WEIGHT, "none")).strip().lower(),
            sgd_learning_rate=str(rc.get(HP_LR_SCHEDULE, "optimal")).strip().lower(),
            sgd_eta0_cfg=float(rc.get(HP_ETA0, 0.0)),
        )

    @staticmethod
    def from_message(message: Message, context: Context) -> "HParams":
        rd: RecordDict = message.content
        cfg: ConfigRecord | None = rd.get("config")
        if cfg is None:
            return HParams.from_run_config(context)

        merged = ConfigRecord(dict(context.run_config))
        for k, v in cfg.items():
            merged[k] = v

        return HParams.from_config(merged)


@dataclass(frozen=True)
class ServerSettings:
    num_rounds: int
    fraction_train: float
    fraction_evaluate: float


def get_server_settings(context: Context) -> ServerSettings:
    return ServerSettings(
        num_rounds=int(context.run_config["num-server-rounds"]),
        fraction_train=float(context.run_config.get("fraction-train", 1.0)),
        fraction_evaluate=float(context.run_config.get("fraction-evaluate", 1.0)),
    )

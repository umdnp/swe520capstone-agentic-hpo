from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import joblib
from dotenv import load_dotenv
from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fedlearn.common.logging_config import setup_logging
from fedlearn.common.model import set_model_params
from fedlearn.hpo.runners import BaselineRunner, StaticHPORunner, AgenticHPORunner, ExperimentRunner

app = ServerApp()

# Constants

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"

RUNNERS: dict[str, Callable[[], ExperimentRunner]] = {
    "baseline": BaselineRunner,
    "static_hpo": StaticHPORunner,
    "agentic_hpo": AgenticHPORunner,
}


@app.main()
def main(grid: Grid, context: Context) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    setup_logging()
    logger = logging.getLogger(__name__)

    # make sure "configs" dir exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    experiment = str(context.run_config.get("experiment", "baseline"))
    factory = RUNNERS.get(experiment)
    if factory is None:
        raise ValueError(f"Unknown experiment {experiment!r}. Valid: {sorted(RUNNERS)}")

    runner = factory()
    result, model = runner.run(grid=grid, context=context)

    # get final global params and save model
    final_params = result.arrays.to_numpy_ndarrays()
    set_model_params(model, final_params)

    save_file = CONFIG_DIR / f"{experiment}.pkl"
    logger.info("Saving final model to %s", save_file)
    joblib.dump(model, save_file)

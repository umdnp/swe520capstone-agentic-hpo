#!/usr/bin/env bash

PROJECT_DIR=~/git
VENV_DIR=~/venv

export PYTHONPATH=$PROJECT_DIR/src
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

source ${VENV_DIR}/activate

echo "Testing Static HPO Configuration ..."
cd $PROJECT_DIR || exit
flwr run . --run-config "experiment=static_hpo"

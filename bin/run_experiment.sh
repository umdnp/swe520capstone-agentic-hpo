#!/usr/bin/env bash
set -euo pipefail

# WARNING : Ray support on Windows is experimental and may not work as expected.
# On Windows, Flower Simulations run best in WSL2

PROJECT_DIR=~/git
VENV_DIR=~/venv

export PYTHONPATH=$PROJECT_DIR/src
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLWR_HOME=$PROJECT_DIR/.flwr

EXPERIMENT="${EXPERIMENT:-baseline}"

source ${VENV_DIR}/activate
cd $PROJECT_DIR || exit 1

cleanup() {
  jobs -p | xargs -r kill
}
trap cleanup EXIT INT TERM

echo "Starting SuperLink ..."
flower-superlink --insecure > superlink.log 2>&1 &
sleep 2

echo "Starting SuperNode 1 ..."
flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config "partition-id=0 num-partitions=3" \
  > supernode-1.log 2>&1 &

echo "Starting SuperNode 2 ..."
flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9095 \
  --node-config "partition-id=1 num-partitions=3" \
  > supernode-2.log 2>&1 &

echo "Starting SuperNode 3 ..."
flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9096 \
  --node-config "partition-id=2 num-partitions=3" \
  > supernode-3.log 2>&1 &

sleep 3

flwr run . local-deployment --stream --run-config "experiment='${EXPERIMENT}'"

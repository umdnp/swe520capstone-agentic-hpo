#!/usr/bin/env bash

PROJECT_DIR=~/git

echo "Testing Static HPO Configuration ..."

EXPERIMENT=static_hpo bash $PROJECT_DIR/bin/run_experiment.sh

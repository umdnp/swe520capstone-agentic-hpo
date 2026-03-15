#!/usr/bin/env bash

PROJECT_DIR=~/git

echo "Testing Agentic HPO Configuration ..."

EXPERIMENT=agentic_hpo bash $PROJECT_DIR/bin/run_experiment.sh

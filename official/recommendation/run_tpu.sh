#!/bin/bash
set -e

export TPU="tayo-tpuv2nightly"
export BUCKET="gs://tayo_datasets"

# Remove IDE "not assigned" warning highlights.
TPU=${TPU:-""}
BUCKET=${BUCKET:-""}

if [[ -z ${TPU} ]]; then
  echo "Please set 'TPU' to the name of the TPU to be used."
  exit 1
fi

if [[ -z ${BUCKET} ]]; then
  echo "Please set 'BUCKET' to the GCS bucket to be used."
  exit 1
fi

./run.sh

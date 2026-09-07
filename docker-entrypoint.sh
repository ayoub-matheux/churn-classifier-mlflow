#!/bin/sh
set -eu

command="$1"
shift

case "$command" in
  train)
    exec python src/train.py --config configs/config.yaml "$@"
    ;;
  evaluate)
    exec python src/evaluate.py --config configs/config.yaml "$@"
    ;;
  predict)
    exec python src/predict.py --config configs/config.yaml "$@"
    ;;
  *)
    echo "Usage: train | evaluate | predict" >&2
    exit 64
    ;;
esac

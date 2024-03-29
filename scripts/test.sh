#!/usr/bin/env bash
#
# Modified from https://github.com/kevinzakka/torchkit/blob/master/scripts/lint.sh
set -xeo pipefail

SRC_FILES=(mission_control/_src/)

if [ "$(uname)" == "Darwin" ]; then
  N_CPU=$(sysctl -n hw.ncpu)
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  N_CPU=$(grep -c ^processor /proc/cpuinfo)
fi

echo "Source format checking"
ruff ${SRC_FILES[@]}
black --check ${SRC_FILES}

if [ "$skipexpensive-false" != "true" ]; then
  echo "Running tests"
  pytest -n "${N_CPU}" --cov-report=term-missing:skip-covered ${SRC_FILES}

  echo "Type checking"
  mypy mission_control
fi
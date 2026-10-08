#!/usr/bin/env bash
# Append one job (a full shell command, run from the project root) to the queue. Usage:
#   jobs/submit.sh 'HIP_VISIBLE_DEVICES=0 .venv/bin/python train.py --models mlp --tag mlp_x'
set -eu
cd "$(dirname "$0")/.."
[ $# -eq 1 ] || { echo "usage: jobs/submit.sh '<command>'"; exit 2; }
flock jobs/queue.lock bash -c 'printf "%s\n" "$1" >> jobs/queue.txt' _ "$1"
echo "queued: $1"
pgrep -f "jobs/runner.sh" >/dev/null || echo "WARNING: runner not running -> jobs/start_runner.sh"

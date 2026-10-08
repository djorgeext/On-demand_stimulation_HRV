#!/usr/bin/env bash
# Start the queue runner fully detached (own session, no terminal): it survives Claude stopping.
cd "$(dirname "$0")/.."
chmod +x jobs/*.sh
setsid nohup jobs/runner.sh > /dev/null 2>&1 < /dev/null &
sleep 1
jobs/status.sh 3

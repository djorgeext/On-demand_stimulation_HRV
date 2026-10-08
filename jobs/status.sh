#!/usr/bin/env bash
# Queue status: runner alive?, current job, queued jobs, last events.
cd "$(dirname "$0")/.."
if pgrep -f "jobs/runner.sh" >/dev/null; then echo "runner: alive"; else echo "runner: NOT RUNNING (jobs/start_runner.sh)"; fi
echo "current: $(cat jobs/current.txt 2>/dev/null || echo none)"
echo "queued ($(grep -c -v -E '^\s*(#|$)' jobs/queue.txt 2>/dev/null)):"
grep -v -E '^\s*(#|$)' jobs/queue.txt 2>/dev/null | sed 's/^/  /'
echo "last events:"
tail -n "${1:-8}" jobs/status.log 2>/dev/null | sed 's/^/  /'

#!/usr/bin/env bash
# Sequential job queue for the dRR project. Runs jobs/queue.txt one line at a time, independent of
# Claude (keeps going through session/usage limits). One job at a time = one GPU job at a time.
# Start:  jobs/start_runner.sh        Submit: jobs/submit.sh '<command>'      Status: jobs/status.sh
# Stop after the current job: touch jobs/STOP
set -u
cd "$(dirname "$0")/.." || exit 1
J=jobs
mkdir -p "$J/logs"
touch "$J/queue.txt" "$J/status.log"
exec 9>"$J/runner.lock"
flock -n 9 || { echo "runner already running"; exit 0; }
log() { echo "$(date '+%F %T') $*" >> "$J/status.log"; }
# the test split is opened once, by the evaluator, interactively: never from the queue
TEST_PAT='subjects_test''_split|hrv_test''\.h5|--split[ =]+te''st'
n=0
log "runner started (pid $$)"
while true; do
  if [ -f "$J/STOP" ]; then rm -f "$J/STOP"; log "runner stopped (STOP file)"; exit 0; fi
  line=$(flock "$J/queue.lock" "$J/_pop.sh")
  if [ -z "$line" ]; then sleep 30; continue; fi
  if echo "$line" | grep -q -E "$TEST_PAT"; then log "REFUSED (test split): $line"; echo "$line" >> "$J/failed.txt"; continue; fi
  n=$((n + 1)); id=$(date "+%m%d_%H%M%S")_$n
  echo "$id $line" > "$J/current.txt"
  log "START $id | $line"
  bash -c "$line" > "$J/logs/$id.log" 2>&1
  rc=$?
  rm -f "$J/current.txt"
  log "END   $id rc=$rc | log jobs/logs/$id.log"
  if [ $rc -eq 0 ]; then echo "$id $line" >> "$J/done.txt"; else echo "$id rc=$rc $line" >> "$J/failed.txt"; fi
done

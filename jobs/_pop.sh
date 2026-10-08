#!/usr/bin/env bash
# Prints and removes the first job line of jobs/queue.txt (skips blanks and # comments).
# Called by runner.sh under flock jobs/queue.lock, so it never races with submit.sh.
cd "$(dirname "$0")/.."
awk 'done || /^[[:space:]]*(#|$)/ {print > "jobs/queue.tmp"; next}
     {job = $0; done = 1}
     END {if (!done) close("jobs/queue.tmp"); printf "%s", job > "/dev/stderr"}' jobs/queue.txt 2> jobs/pop.out
touch jobs/queue.tmp
mv jobs/queue.tmp jobs/queue.txt
cat jobs/pop.out
rm -f jobs/pop.out

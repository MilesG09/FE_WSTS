#!/bin/bash
# Generic chaining utility: blocks until the given PID exits, then execs the given command.
# For running a second job unattended, immediately after a first one finishes, without anyone
# needing to be present to notice completion and launch the next step by hand.
#
# Usage: wait_and_chain.sh <pid> <command...>
set -uo pipefail

PID=${1:?pid required}
shift

echo "=== [$(date)] Waiting for pid $PID to exit before starting: $* ==="
while kill -0 "$PID" 2>/dev/null; do
    sleep 30
done
echo "=== [$(date)] pid $PID exited -- starting chained command ==="

exec "$@"

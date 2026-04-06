#!/bin/bash
# Memory watchdog — kills user Python processes if RAM usage exceeds 98%
# Usage: bash watchdog.sh &

THRESHOLD=98

while true; do
    usage=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
    if [ "$usage" -ge "$THRESHOLD" ]; then
        echo "$(date): Memory at ${usage}% — killing user Python processes"
        pkill -u "$(whoami)" -f python --signal KILL
        sleep 5
    fi
    sleep 2
done

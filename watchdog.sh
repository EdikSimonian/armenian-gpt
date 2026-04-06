#!/bin/bash
# Memory watchdog — kills user Python processes if RAM usage exceeds 98%
# Usage: nohup bash watchdog.sh > watchdog.log 2>&1 &

THRESHOLD=98
LOG="/workspace/armenian-gpt/watchdog.log"

while true; do
    usage=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
    if [ "$usage" -ge "$THRESHOLD" ]; then
        echo "========================================" >> "$LOG"
        echo "$(date): ALERT — Memory at ${usage}%" >> "$LOG"
        echo "Memory before kill:" >> "$LOG"
        free -h >> "$LOG"
        echo "" >> "$LOG"
        echo "Processes to kill:" >> "$LOG"
        ps aux --sort=-%mem | grep -E 'python|Python' | grep -v grep >> "$LOG"
        echo "" >> "$LOG"
        pkill -u "$(whoami)" -f python --signal KILL
        echo "$(date): Killed all user Python processes" >> "$LOG"
        sleep 5
        echo "Memory after kill:" >> "$LOG"
        free -h >> "$LOG"
        echo "========================================" >> "$LOG"
    fi
    sleep 2
done

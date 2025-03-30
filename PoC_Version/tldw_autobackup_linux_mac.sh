#!/bin/bash

# Path to your Litestream configuration
LITESTREAM_CONFIG="litestream.yml"

# Command to start your application
APP_COMMAND="python3 summarize.py -gui -log INFO"

# Log file path
LOG_FILE="tldw_backup.log"

# Start Litestream with restore flags
litestream replicate \
  -config "$LITESTREAM_CONFIG" \
  -exec "$APP_COMMAND" \
  -restore -if-replica-exists \
  -v 2>&1 | tee -a "$LOG_FILE"

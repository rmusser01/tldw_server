#!/bin/bash

# TLDW Run Script for macOS

install_dir="$(dirname "$0")/tldw"

if [ ! -d "$install_dir" ]; then
    echo "TLDW installation not found. Please run the install_update_tldw.sh script first."
    exit 1
fi

cd "$install_dir" || exit

# Activate virtual environment
source venv/bin/activate

# Run TLDW
python3 summarize.py -gui

# Deactivate virtual environment when done
deactivate
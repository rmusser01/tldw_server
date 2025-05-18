#!/bin/bash

# TLDW macOS Installer and Updater Script

# Base directory where this script is located
script_base_dir="$(cd "$(dirname "$0")" && pwd)"
log_file="$script_base_dir/tldw_install_log.txt"

# Repository and target application directory names
repo_name="tldw"
app_subdir="PoC_Version" # New: Subdirectory for the application code

# Full path to the cloned repository and the application directory
install_dir="$script_base_dir/$repo_name"
app_install_dir="$install_dir/$app_subdir"

# --- Logging ---
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" >> "$log_file"
}

log "--- Starting TLDW installation/update process for macOS ---"

# --- Helper Functions ---
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

ask_permission() {
    read -r -p "Would you like to install $1? (y/n) [y]: " choice # Added default y
    choice=${choice:-y}
    case "$choice" in
        y|Y ) return 0;;
        n|N ) return 1;;
        * ) echo "Invalid input. Please enter 'y' or 'n'."; ask_permission "$1";;
    esac
}

install_homebrew() {
    log "Checking for Homebrew..."
    if ! command_exists brew; then
        echo "Homebrew not found."
        if ask_permission "Homebrew (recommended for managing dependencies)"; then
            log "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            if [ $? -ne 0 ]; then
                log "Homebrew installation failed."
                echo "Error: Homebrew installation failed. Please try installing it manually."
                exit 1
            fi
            # Attempt to add Homebrew to PATH for the current session
            if [ -x "/opt/homebrew/bin/brew" ]; then
                export PATH="/opt/homebrew/bin:$PATH"
                log "Temporarily added /opt/homebrew/bin to PATH for M1/M2 Macs."
            elif [ -x "/usr/local/bin/brew" ]; then
                 export PATH="/usr/local/bin:$PATH"
                 log "Temporarily added /usr/local/bin to PATH for Intel Macs."
            fi
            log "Homebrew installed."
        else
            log "Homebrew installation skipped by user."
            echo "Homebrew installation skipped. Some dependencies might need manual installation."
        fi
    else
        log "Homebrew is already installed."
    fi
}

install_package_brew() {
    local package_name="$1"
    if ! command_exists brew; then
        echo "Homebrew is not available. Cannot install $package_name automatically."
        log "Homebrew not available for $package_name."
        return 1 # Indicate failure
    fi
    if ! command_exists "$package_name"; then
        echo "$package_name not found."
        if ask_permission "$package_name (via Homebrew)"; then
            log "Installing $package_name using Homebrew..."
            if brew install "$package_name"; then
                log "$package_name installed successfully via Homebrew."
            else
                log "Failed to install $package_name via Homebrew."
                echo "Error: Failed to install $package_name using Homebrew. Please try manually."
                return 1
            fi
        else
            log "$package_name installation skipped by user."
            echo "$package_name installation skipped."
            return 1 # Indicate skipped/failed
        fi
    else
        log "$package_name is already installed."
    fi
    return 0
}


# --- Update Logic ---
update_installation() {
    log "Starting update for existing installation in $install_dir"
    cd "$install_dir" || { log "Failed to cd into $install_dir"; echo "Error: Could not navigate to $install_dir"; exit 1; }

    log "Fetching remote changes..."
    if ! git fetch; then
        log "git fetch failed."
        echo "Error: git fetch failed. Check your internet connection or git configuration."
        exit 1
    fi

    local old_version
    old_version=$(git rev-parse HEAD)
    local new_version
    new_version=$(git rev-parse @{u})

    if [ "$old_version" == "$new_version" ]; then
        echo "TLDW is already up to date (Version: $old_version)."
        log "TLDW is already up to date."
        read -r -p "Do you want to re-check/re-install requirements anyway? (y/n) [n]: " reinstall_reqs
        reinstall_reqs=${reinstall_reqs:-n}
        if [[ "$reinstall_reqs" =~ ^[Yy]$ ]] || [ ! -d "$app_install_dir/venv" ]; then
            echo "Setting up/checking environment..."
            setup_environment
        else
            echo "To re-initialize the environment, consider removing the 'venv' directory from '$app_install_dir' and re-running the installer."
        fi
        return 0
    fi

    echo "Current version: $old_version"
    echo "New version available: $new_version"
    read -r -p "Do you want to proceed with the update? (y/n): " confirm_update
    if [[ "$confirm_update" =~ ^[Yy]$ ]]; then
        log "User confirmed update. Creating backup..."
        backup_name="${repo_name}_backup_$(date +%Y%m%d_%H%M%S)"
        if cp -R "$install_dir" "$script_base_dir/$backup_name"; then
            log "Backup created at $script_base_dir/$backup_name"
        else
            log "Failed to create backup. Continuing without backup."
            echo "Warning: Failed to create backup."
        fi

        log "Pulling latest changes..."
        if ! git pull; then
            log "git pull failed."
            echo "Error: git pull failed. Please check for conflicts or stash local changes."
            exit 1
        fi
        log "Successfully pulled latest changes."
        echo "TLDW updated successfully."
        setup_environment
    else
        log "Update cancelled by user."
        echo "Update cancelled."
    fi
}


# --- Fresh Install Logic ---
fresh_installation() {
    log "Starting fresh installation into $install_dir"

    if [ -d "$install_dir" ]; then
        read -r -p "Directory '$install_dir' already exists. Do you want to remove it and proceed with a fresh install? (y/n): " remove_confirm
        if [[ "$remove_confirm" =~ ^[Yy]$ ]]; then
            log "Removing existing directory: $install_dir"
            rm -rf "$install_dir"
            if [ $? -ne 0 ]; then
                log "Failed to remove existing directory $install_dir"
                echo "Error: Failed to remove existing directory. Please remove it manually."
                exit 1
            fi
        else
            log "Fresh installation aborted by user due to existing directory."
            echo "Installation aborted. Directory '$install_dir' was not removed."
            exit 0
        fi
    fi

    mkdir -p "$install_dir" # Ensure base install_dir exists

    read -r -p "Do you want to attempt GPU (Apple Metal/MPS) accelerated PyTorch installation? (y/n) [y]: " gpu_support
    gpu_support=${gpu_support:-y} # Default to 'y' for macOS as MPS is common

    local gpu_choice="cpu" # Default, will be updated if MPS is chosen and confirmed
    if [[ "$gpu_support" =~ ^[Yy]$ ]]; then
        echo "PyTorch on macOS can use Apple's Metal Performance Shaders (MPS) for GPU acceleration."
        echo "The standard PyTorch installation will attempt to use MPS if available."
        echo "No special CUDA/ROCm drivers are needed."
        gpu_choice="mps" # Indicate intent for MPS, PyTorch handles the rest
    else
        echo "Proceeding with CPU-only PyTorch installation."
        gpu_choice="cpu"
    fi
    log "GPU preference set to: $gpu_choice (PyTorch will use MPS if available and 'mps' selected, otherwise CPU)"

    log "Cloning repository https://github.com/rmusser01/tldw into $install_dir"
    if ! git clone https://github.com/rmusser01/tldw "$install_dir"; then
        log "git clone failed."
        echo "Error: Could not clone the repository. Check your internet connection and git installation."
        exit 1
    fi
    log "Repository cloned successfully."

    # Create gpu_choice.txt *after* cloning and *before* setup_environment
    # Ensure the $app_install_dir exists before writing to it
    mkdir -p "$app_install_dir"
    echo "$gpu_choice" > "$app_install_dir/gpu_choice.txt"
    log "Saved GPU choice to $app_install_dir/gpu_choice.txt"

    setup_environment
}


# --- Environment Setup ---
setup_environment() {
    log "Navigating to application directory: $app_install_dir"
    cd "$app_install_dir" || { log "Failed to cd into $app_install_dir"; echo "Error: Could not navigate to $app_install_dir"; exit 1; }

    log "Setting up Python virtual environment in $app_install_dir/venv"
    if [ ! -d "venv" ]; then
        if ! python3 -m venv ./venv; then
            log "Failed to create virtual environment."
            echo "Error: Failed to create Python virtual environment."
            exit 1
        fi
        log "Virtual environment created."
    else
        log "Virtual environment already exists."
    fi

    log "Activating virtual environment."
    # shellcheck source=/dev/null
    source ./venv/bin/activate
    if [ $? -ne 0 ]; then
        log "Failed to activate virtual environment."
        echo "Error: Failed to activate Python virtual environment."
        exit 1
    fi

    log "Upgrading pip and installing wheel."
    if ! python -m pip install --upgrade pip wheel; then
        log "Failed to upgrade pip or install wheel."
        echo "Error: Failed to upgrade pip or install wheel."
        deactivate
        exit 1
    fi

    # Read GPU choice
    local local_gpu_choice="cpu" # Default if file is missing
    if [ -f "$app_install_dir/gpu_choice.txt" ]; then
        local_gpu_choice=$(cat "$app_install_dir/gpu_choice.txt")
    else
        log "gpu_choice.txt not found in $app_install_dir during setup. Defaulting to CPU."
    fi
    log "GPU choice for PyTorch installation: $local_gpu_choice"

    # Ensure config.txt exists before trying to modify it.
    if [ ! -f "Config_Files/config.txt" ]; then
        log "Config_Files/config.txt not found. Creating a default."
        mkdir -p Config_Files
        echo "[Processing]" > Config_Files/config.txt
        echo "processing_choice = $local_gpu_choice" >> Config_Files/config.txt # Use local_gpu_choice
    fi

    log "Installing PyTorch (macOS version, will use MPS if available and selected)..."
    # For macOS, `pip install torch torchvision torchaudio` is usually sufficient.
    # It will pull a build that supports MPS if your Mac hardware/OS does.
    if ! python -m pip install torch torchvision torchaudio; then
        log "Failed to install PyTorch."
        echo "Error: Failed to install PyTorch. Check your network and Python environment."
        deactivate
        exit 1
    fi
    log "PyTorch installed successfully."

    # Update config.txt based on the choice (though MPS is often auto-detected by PyTorch)
    # The config setting 'mps' or 'cpu' might be used by your app to explicitly set the device.
    if [ "$local_gpu_choice" == "mps" ]; then
        sed -i '' 's/processing_choice = .*/processing_choice = mps/' Config_Files/config.txt
        log "Updated config.txt processing_choice to mps."
    else # cpu
        sed -i '' 's/processing_choice = .*/processing_choice = cpu/' Config_Files/config.txt
        log "Updated config.txt processing_choice to cpu."
    fi
    # Remove backup file created by sed -i on macOS
    [ -f "Config_Files/config.txt" ] && rm -f "Config_Files/config.txt''"


    log "Installing other requirements from requirements.txt"
    if [ -f "requirements.txt" ]; then
        if ! python -m pip install -r requirements.txt; then
            log "Failed to install packages from requirements.txt"
            echo "Error: Failed to install required packages from requirements.txt. Check the file and your network."
            deactivate
            exit 1
        fi
        log "Successfully installed packages from requirements.txt"
    else
        log "requirements.txt not found in $app_install_dir. Skipping."
        echo "Warning: requirements.txt not found. Some features might not work."
    fi

    log "Environment setup complete."
}


# --- Main Script Execution ---
install_homebrew # Check/install Homebrew first as other packages might depend on it

log "Checking prerequisite packages: python3, git, ffmpeg"
# Check python3 again in case Homebrew installed/updated it
if ! command_exists python3; then
    echo "python3 not found. Attempting to install via Homebrew..."
    install_package_brew "python3" || { echo "Critical: python3 could not be installed."; exit 1; }
else
    log "python3 is already installed."
fi

if ! command_exists git; then
    echo "git not found. Attempting to install via Homebrew..."
    install_package_brew "git" || { echo "Warning: git could not be installed. Manual installation might be needed."; }
else
    log "git is already installed."
fi

if ! command_exists ffmpeg; then
    echo "ffmpeg not found. Attempting to install via Homebrew..."
    install_package_brew "ffmpeg" || { echo "Warning: ffmpeg could not be installed. Manual installation might be needed for full functionality."; }
else
    log "ffmpeg is already installed."
fi
log "Prerequisite check finished."


if [ -d "$install_dir/.git" ]; then
    read -r -p "TLDW repository found in '$install_dir'. Do you want to update it? (y/n) [y]: " update_choice
    update_choice=${update_choice:-y}
    if [[ "$update_choice" =~ ^[Yy]$ ]]; then
        update_installation
    else
        log "User chose not to update existing installation."
        echo "Skipping update. To do a fresh install, remove the '$install_dir' directory and re-run."
        if [ ! -d "$app_install_dir/venv" ]; then
            echo "Virtual environment not found in $app_install_dir. Setting it up..."
            setup_environment
        fi
    fi
else
    fresh_installation
fi

log "--- TLDW installation/update process completed for macOS ---"
echo ""
echo "Installation/Update completed!"

read -r -p "Do you want to run TLDW now? (y/n) [y]: " run_now_choice
run_now_choice=${run_now_choice:-y}
if [[ "$run_now_choice" =~ ^[Yy]$ ]]; then
    echo "Attempting to run TLDW..."
    cd "$app_install_dir" || { log "Failed to cd into $app_install_dir to run script."; echo "Error: Could not navigate to $app_install_dir to run."; exit 1; }

    if [ ! -f "summarize.py" ]; then
        log "summarize.py not found in $app_install_dir."
        echo "Error: summarize.py not found. The application structure might be incorrect."
        exit 1
    fi

    log "Activating venv to run summarize.py"
    # shellcheck source=/dev/null
    source ./venv/bin/activate
    if [ $? -ne 0 ]; then
        log "Failed to activate venv for running the script."
        echo "Error: Failed to activate venv."
        exit 1
    fi

    log "Running: python3 summarize.py -gui -log DEBUG"
    echo "Starting TLDW GUI... (Press Ctrl+C to exit)"
    python3 summarize.py -gui -log DEBUG

    log "Deactivating venv."
    deactivate
    echo "TLDW application finished."
else
    echo "You can run TLDW later by navigating to '$app_install_dir', activating the environment ('source venv/bin/activate'), and then running 'python3 summarize.py -gui'."
fi

echo "Script finished. Check '$log_file' for details."
#!/bin/bash

# TLDW Installation and Update Script

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

log "--- Starting TLDW installation/update process ---"

# --- Helper Functions ---
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_package() {
    local package_name="$1"
    log "Attempting to install $package_name"
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y "$package_name"
    elif command_exists dnf; then
        sudo dnf install -y "$package_name"
    elif command_exists yum; then # Added yum for older RHEL/CentOS
        sudo yum install -y "$package_name"
    elif command_exists pacman; then # Added pacman for Arch
        sudo pacman -Syu --noconfirm "$package_name"
    else
        echo "Unsupported package manager. Please install $package_name manually."
        log "Unsupported package manager for $package_name. Installation aborted."
        exit 1
    fi
    if ! command_exists "$package_name"; then
        log "Failed to install $package_name."
        echo "Error: Failed to install $package_name. Please check the logs and try manually."
        exit 1
    fi
    log "$package_name installed successfully."
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
    new_version=$(git rev-parse @{u}) # origin/main or origin/master

    if [ "$old_version" == "$new_version" ]; then
        echo "TLDW is already up to date (Version: $old_version)."
        log "TLDW is already up to date."
        # Optionally, still offer to re-setup environment if requested or if venv is missing
        if [ ! -d "$app_install_dir/venv" ]; then
            echo "Virtual environment not found. Setting it up..."
            setup_environment # Call setup_environment which cds into app_install_dir
        else
            echo "To re-initialize the environment, consider removing the 'venv' directory and re-running the installer."
        fi
        return 0 # Use return for functions, exit for script termination
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
        setup_environment # Call setup_environment which cds into app_install_dir
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

    mkdir -p "$install_dir" # Ensure base install_dir exists before trying to save gpu_choice.txt

    read -r -p "Do you want to install with GPU support? (y/n) [n]: " gpu_support
    gpu_support=${gpu_support:-n} # Default to 'n' if user presses Enter

    local gpu_choice="cpu"
    if [[ "$gpu_support" =~ ^[Yy]$ ]]; then
        echo "Select GPU type:"
        echo "  1) NVIDIA (CUDA)"
        echo "  2) AMD (ROCm/DirectML - DirectML preferred for general PyTorch on Windows, ROCm for Linux)"
        echo "  *) CPU (default)"
        read -r -p "Enter choice [CPU]: " gpu_type_choice
        gpu_type_choice=${gpu_type_choice:-CPU}

        case "$gpu_type_choice" in
            1|[Nn][Vv][Ii][Dd][Ii][Aa]|[Cc][Uu][Dd][Aa])
                echo "Configuring for NVIDIA CUDA support."
                echo "Please ensure your NVIDIA drivers and CUDA Toolkit (version compatible with PyTorch) are installed."
                echo "Refer to: https://developer.nvidia.com/cuda-downloads and https://pytorch.org/get-started/locally/"
                gpu_choice="cuda"
                ;;
            2|[Aa][Mm][Dd]|[Rr][Oo][Cc][Mm])
                echo "Configuring for AMD GPU support (ROCm for Linux)."
                echo "Please ensure your AMD drivers and ROCm are installed and configured."
                echo "Refer to: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html and https://pytorch.org/get-started/locally/"
                gpu_choice="amd" # For Linux, this implies ROCm for PyTorch
                ;;
            *)
                echo "Invalid choice or no choice. Defaulting to CPU installation."
                gpu_choice="cpu"
                ;;
        esac
    else
        echo "Proceeding with CPU-only installation."
        gpu_choice="cpu"
    fi

    log "GPU choice set to: $gpu_choice"
    # Save GPU choice inside the PoC_Version directory *after* cloning
    # echo "$gpu_choice" > "$app_install_dir/gpu_choice.txt" # This will be done in setup_environment

    log "Cloning repository https://github.com/rmusser01/tldw into $install_dir"
    if ! git clone https://github.com/rmusser01/tldw "$install_dir"; then
        log "git clone failed."
        echo "Error: Could not clone the repository. Check your internet connection and git installation."
        exit 1
    fi
    log "Repository cloned successfully."

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

    # Read GPU choice (assuming it was set during fresh_install or already exists for update)
    # For fresh install, we create it here. For update, it should exist or default to CPU.
    if [ -f "$app_install_dir/gpu_choice.txt" ]; then
        gpu_choice=$(cat "$app_install_dir/gpu_choice.txt")
    else
        log "gpu_choice.txt not found, defaulting to CPU or relying on previous choice if update."
        if [ ! -f "$app_install_dir/gpu_choice.txt" ]; then
            log "gpu_choice.txt not found in $app_install_dir during setup for update. Defaulting to CPU."
            echo "cpu" > "$app_install_dir/gpu_choice.txt"
        fi
        gpu_choice=$(cat "$app_install_dir/gpu_choice.txt")

    fi
    log "GPU choice for PyTorch installation: $gpu_choice"

    # Ensure config.txt exists before trying to modify it.
    # Create a default one if it's missing from the repo (should not happen).
    if [ ! -f "Config_Files/config.txt" ]; then
        log "Config_Files/config.txt not found. Creating a default or placeholder."
        mkdir -p Config_Files
        # Create a minimal config or copy a template if you have one
        echo "[Processing]" > Config_Files/config.txt
        echo "processing_choice = $gpu_choice" >> Config_Files/config.txt
    fi

    log "Installing PyTorch ($gpu_choice version)..."
    # PyTorch CUDA 12.1 is compatible with cu123 for PyTorch 2.2.x
    # The index URL for cu123 should work for PyTorch 2.2.2 if it includes CUDA 12.1 builds
    if [ "$gpu_choice" == "cuda" ]; then
        # Using cu121 for PyTorch 2.2.2 as it's a common stable combination. Adjust if needed.
        if ! python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121; then
            log "Failed to install PyTorch with CUDA. Check CUDA/driver compatibility or network."
            echo "Error: Failed to install PyTorch with CUDA. Attempting CPU version..."
            if ! python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu; then
                 log "Failed to install PyTorch CPU version."
                 echo "Error: Failed to install PyTorch CPU version."
                 deactivate
                 exit 1
            fi
            log "PyTorch CPU version installed as fallback."
            # Update config.txt to use CPU if CUDA install failed
            sed -i.bak 's/processing_choice = cuda/processing_choice = cpu/' Config_Files/config.txt && rm Config_Files/config.txt.bak
            log "Updated config.txt to use CPU due to PyTorch CUDA installation failure."
        else
             log "PyTorch with CUDA installed successfully."
             sed -i.bak 's/processing_choice = cpu/processing_choice = cuda/' Config_Files/config.txt && rm Config_Files/config.txt.bak
             log "Ensured config.txt is set to CUDA."
        fi
    elif [ "$gpu_choice" == "amd" ]; then
        # For Linux, AMD usually means ROCm for PyTorch
        if ! python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7; then # Adjust ROCm version if needed
            log "Failed to install PyTorch with ROCm. Check ROCm installation or network."
            echo "Error: Failed to install PyTorch with ROCm. Attempting CPU version..."
            if ! python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu; then
                log "Failed to install PyTorch CPU version."
                echo "Error: Failed to install PyTorch CPU version."
                deactivate
                exit 1
            fi
            log "PyTorch CPU version installed as fallback."
            sed -i.bak 's/processing_choice = .*/processing_choice = cpu/' Config_Files/config.txt && rm Config_Files/config.txt.bak
            log "Updated config.txt to use CPU due to PyTorch ROCm installation failure."
        else
            log "PyTorch with ROCm installed successfully."
            sed -i.bak 's/processing_choice = cpu/processing_choice = amdgpu/' Config_Files/config.txt && rm Config_Files/config.txt.bak # Or 'rocm'
            log "Ensured config.txt is set to AMD."
        fi
    else # CPU
        if ! python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu; then
             log "Failed to install PyTorch CPU version."
             echo "Error: Failed to install PyTorch CPU version."
             deactivate
             exit 1
        fi
        log "PyTorch CPU version installed successfully."
        # Update config.txt to use CPU
        sed -i.bak 's/processing_choice = .*/processing_choice = cpu/' Config_Files/config.txt && rm Config_Files/config.txt.bak
        log "Updated config.txt to use CPU."
    fi

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
    # Deactivation will happen after running the script or explicitly
}


# --- Main Script Execution ---
log "Checking prerequisite packages: python3, git, ffmpeg"
for package in python3 git ffmpeg; do
    if ! command_exists "$package"; then
        echo "$package not found. Attempting to install..."
        install_package "$package"
    else
        log "$package is already installed."
    fi
done
log "All prerequisite packages are installed."

# Check if this is an update or new installation
if [ -d "$install_dir/.git" ]; then # Check for .git to confirm it's a repo
    read -r -p "TLDW repository found in '$install_dir'. Do you want to update it? (y/n) [y]: " update_choice
    update_choice=${update_choice:-y}
    if [[ "$update_choice" =~ ^[Yy]$ ]]; then
        update_installation
    else
        log "User chose not to update existing installation."
        echo "Skipping update. To do a fresh install, remove the '$install_dir' directory and re-run."
        # Optionally, still offer to re-setup environment if requested
        if [ ! -d "$app_install_dir/venv" ]; then
            echo "Virtual environment not found. Setting it up..."
            setup_environment
        fi
    fi
else
    fresh_installation
fi

log "--- TLDW installation/update process completed ---"
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
    python3 summarize.py -gui -log DEBUG # Or your desired log level for running

    log "Deactivating venv."
    deactivate
    echo "TLDW application finished."
else
    echo "You can run TLDW later by navigating to '$app_install_dir' and running 'source venv/bin/activate' then 'python3 summarize.py -gui'."
fi

echo "Script finished. Check '$log_file' for details."
#!/bin/bash

# TLDW Installation and Update Script

# Set up logging
log_file="$(dirname "$0")/tldw_install_log.txt"
log() {
    echo "$(date): $1" >> "$log_file"
}

log "Starting TLDW installation/update process"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages based on the package manager
install_package() {
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y "$1"
    elif command_exists dnf; then
        sudo dnf install -y "$1"
    else
        echo "Unsupported package manager. Please install $1 manually."
        log "Unsupported package manager for $1"
        exit 1
    fi
}

update() {
    log "Updating existing installation"
    cd "$install_dir" || exit
    git fetch
    old_version=$(git rev-parse HEAD)
    new_version=$(git rev-parse @{u})
    if [ "$old_version" == "$new_version" ]; then
        echo "TLDW is already up to date."
        log "TLDW is already up to date"
        exit 0
    fi
    echo "Current version: $old_version"
    echo "New version: $new_version"
    read -p "Do you want to proceed with the update? (y/n): " confirm_update
    if [[ $confirm_update == "y" || $confirm_update == "Y" ]]; then
        log "Creating backup"
        cp -R "$install_dir" "${install_dir}_backup_$(date +%Y%m%d)"
        if ! git pull; then
            log "Git pull failed"
            echo "Error: Git pull failed. Please check your internet connection and try again."
            exit 1
        fi
    else
        log "Update cancelled by user"
        echo "Update cancelled."
        exit 0
    fi
    setup_environment
}

fresh_install() {
    log "Starting fresh installation"
    # Prompt for GPU installation
    read -p "Do you want to install with GPU support? (y/n): " gpu_support
    if [[ $gpu_support == "y" || $gpu_support == "Y" ]]; then
        read -p "Choose GPU type (1 for CUDA, 2 for AMD): " gpu_type
        if [ "$gpu_type" == "1" ]; then
            echo "Please ensure your NVIDIA GPU drivers and CUDA are up to date."
            echo "Visit https://developer.nvidia.com/cuda-downloads for instructions."
            gpu_choice="cuda"
        elif [ "$gpu_type" == "2" ]; then
            echo "Please ensure your AMD GPU drivers are up to date."
            gpu_choice="amd"
        else
            echo "Invalid choice. Defaulting to CPU installation."
            gpu_choice="cpu"
        fi
    else
        gpu_choice="cpu"
    fi

    # Save GPU choice
    echo "$gpu_choice" > "$install_dir/gpu_choice.txt"

    # Clone the repository
    git clone https://github.com/rmusser01/tldw "$install_dir"
    cd "$install_dir" || exit

    setup_environment
}

setup_environment() {
    log "Setting up environment"
    # Create and activate virtual environment (if it doesn't exist)
    if [ ! -d "venv" ]; then
        python3 -m venv ./venv
    fi
    source ./venv/bin/activate

    # Upgrade pip and install wheel
    pip install --upgrade pip wheel

    # Read GPU choice
    gpu_choice=$(cat "$install_dir/gpu_choice.txt")

    # Install PyTorch based on GPU support choice
    if [ "$gpu_choice" == "cuda" ]; then
        pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu123
    elif [ "$gpu_choice" == "amd" ]; then
        pip install torch-directml
    else
        pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
        # Update config.txt to use CPU
        sed -i 's/cuda/cpu/' config.txt
    fi

    # Install other requirements
    pip install -r requirements.txt
}

# Main script execution

# Check and install required packages
for package in python3 git ffmpeg; do
    if ! command_exists $package; then
        echo "$package not found. Installing..."
        log "Installing $package"
        install_package $package
    fi
done

install_dir="$(dirname "$0")/tldw"

# Check if this is an update or new installation
if [ -d "$install_dir" ]; then
    read -p "TLDW directory found. Do you want to update? (y/n): " update_choice
    if [[ $update_choice == "y" || $update_choice == "Y" ]]; then
        update
    else
        fresh_install
    fi
else
    fresh_install
fi

log "Installation/Update process completed"
echo "Installation/Update completed successfully!"
echo "To run TLDW, use the run_tldw.sh script"
echo "Which is what I'm doing for you now..."
# Run TLDW
cd "tldw" || exit
source venv/bin/activate
python3 summarize.py -gui
deactivate
echo "TLDW has been ran. Goodbye!"
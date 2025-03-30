#!/bin/bash

# TLDW macOS Installer and Updater Script

# Set up logging
log_file="$(dirname "$0")/tldw_install_log.txt"
log() {
    echo "$(date): $1" >> "$log_file"
}

log "Starting TLDW installation/update process for macOS"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to ask for permission to install
ask_permission() {
    read -p "Would you like to install $1? (y/n): " choice
    case "$choice" in
        y|Y ) return 0;;
        n|N ) return 1;;
        * ) echo "Invalid input. Please enter 'y' or 'n'."; ask_permission "$1";;
    esac
}

# Function to install Homebrew
install_homebrew() {
    if ask_permission "Homebrew"; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "Homebrew installation skipped. Some features may not work without it."
        log "Homebrew installation skipped by user"
    fi
}

# Function to install packages using Homebrew
install_package() {
    if ! command_exists brew; then
        echo "Homebrew is required to install $1."
        install_homebrew
    fi
    if command_exists brew; then
        if ask_permission "$1"; then
            brew install "$1"
        else
            echo "$1 installation skipped. Some features may not work without it."
            log "$1 installation skipped by user"
        fi
    fi
}

# Check and install required packages
for package in python3 git ffmpeg; do
    if ! command_exists $package; then
        echo "$package is required for this installation."
        install_package $package
    fi
done

install_dir="$(dirname "$0")/tldw"

# Functions

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
        echo "Note: GPU support on macOS is limited. Defaulting to CPU installation."
        gpu_choice="cpu"
    else
        gpu_choice="cpu"
    fi

    # Clone the repository
    git clone https://github.com/rmusser01/tldw "$install_dir"

    # Save GPU choice
    mkdir -p "$install_dir"
    echo "$gpu_choice" > "$install_dir/gpu_choice.txt"

    # Move into the installation directory and set up the environment
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

    # Install PyTorch (CPU version for macOS)
    pip install torch torchvision torchaudio

    # Update config.txt to use CPU
    sed -i '' 's/cuda/cpu/' config.txt

    # Install requirements from the cloned repository
    if [ -f "requirements.txt" ]; then
        log "Installing requirements from requirements.txt"
        pip install -r requirements.txt
    else
        log "requirements.txt not found in the installation directory"
        echo "Warning: requirements.txt not found. Some dependencies may be missing."
    fi
}

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
echo "To run TLDW, use the run_tldw.sh script"
echo "Which is what I'm doing for you now..."
# Run TLDW
cd "$install_dir" || exit
source venv/bin/activate
python3 summarize.py -gui
deactivate
echo "TLDW has been ran. Goodbye!"
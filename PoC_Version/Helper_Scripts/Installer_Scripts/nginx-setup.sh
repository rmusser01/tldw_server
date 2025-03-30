#!/bin/bash

# Nginx Reverse Proxy Setup Script for TLDW

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to display a prominent message
display_message() {
    echo ""
    echo "**********************************************"
    echo "$1"
    echo "**********************************************"
    echo ""
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect the operating system
if command_exists apt-get; then
    log "Debian/Ubuntu system detected"
    # Debian/Ubuntu
    log "Updating package lists..."
    sudo apt update
    log "Installing Nginx..."
    sudo apt install -y nginx
elif command_exists yum; then
    log "CentOS/RHEL system detected"
    # CentOS/RHEL
    log "Installing EPEL repository..."
    sudo yum install -y epel-release
    log "Installing Nginx..."
    sudo yum install -y nginx
else
    log "Unsupported operating system"
    exit 1
fi

# Start and enable Nginx
log "Starting and enabling Nginx..."
sudo systemctl start nginx
sudo systemctl enable nginx

# Create a new configuration file
log "Creating Nginx configuration file..."
sudo tee /etc/nginx/sites-available/tldw_app > /dev/null << EOL
server {
    listen 80;
    server_name your_server_ip_or_domain;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOL

# Enable the site
log "Enabling the Nginx site..."
sudo ln -s /etc/nginx/sites-available/tldw_app /etc/nginx/sites-enabled/

# Test Nginx configuration
log "Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    # Reload Nginx
    log "Reloading Nginx..."
    sudo systemctl reload nginx

    # Configure firewall if ufw is available
    if command_exists ufw; then
        log "Configuring firewall..."
        sudo ufw allow 80/tcp
    else
        log "UFW firewall not found. Please configure your firewall manually to allow port 80/tcp."
    fi

    log "Nginx reverse proxy setup completed successfully!"

    # Display prominent message about updating server_name
    display_message "IMPORTANT: You must update the server_name in the Nginx configuration!

    Please edit the file /etc/nginx/sites-available/tldw_app
    Replace 'your_server_ip_or_domain' with your actual server IP or domain name.

    After editing, run: sudo nginx -t && sudo systemctl reload nginx"

else
    log "Nginx configuration test failed. Please check your configuration and try again."
    exit 1
fi
@echo off
setlocal enabledelayedexpansion

:: TLDW Windows Installer and Updater Script

:: Set up logging
set "log_file=%~dp0tldw_install_log.txt"
call :log "Starting TLDW installation/update process"

:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    call :log "Python is not installed or not in PATH."
    echo Python is not installed or not in PATH.
    echo Please download and install Python from https://www.python.org/downloads/windows/
    echo After installation, run this script again.
    pause
    exit /b 1
)

:: Check for git installation
git --version >nul 2>&1
if %errorlevel% neq 0 (
    call :log "Git is not installed or not in PATH."
    echo Git is not installed or not in PATH.
    echo Please download and install Git from https://git-scm.com/download/win
    echo After installation, run this script again.
    pause
    exit /b 1
)

:: Check if this is an update or new installation
set "install_dir=%~dp0tldw"
if exist "%install_dir%" (
    set /p update_choice="TLDW directory found. Do you want to update? (y/n): "
    if /i "!update_choice!"=="y" (
        call :update
    ) else (
        call :fresh_install
    )
) else (
    call :fresh_install
)

call :cleanup
call :log "Installation/Update process completed"
echo Installation/Update completed successfully!
echo To run TLDW, use the run_tldw.bat script.
pause
exit /b 0

:update
call :log "Updating existing installation"
cd "%install_dir%"
git fetch
for /f %%i in ('git rev-parse HEAD') do set "old_version=%%i"
for /f %%i in ('git rev-parse @{u}') do set "new_version=%%i"
if "%old_version%"=="%new_version%" (
    echo TLDW is already up to date.
    call :log "TLDW is already up to date"
    exit /b 0
)
echo Current version: %old_version%
echo New version: %new_version%
set /p confirm_update="Do you want to proceed with the update? (y/n): "
if /i "!confirm_update!"=="y" (
    call :log "Creating backup"
    xcopy /E /I /H /Y "%install_dir%" "%install_dir%_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%"
    git pull
    if %errorlevel% neq 0 (
        call :log "Git pull failed"
        echo Error: Git pull failed. Please check your internet connection and try again.
        exit /b 1
    )
) else (
    call :log "Update cancelled by user"
    echo Update cancelled.
    exit /b 0
)
call :setup_environment
goto :eof

:fresh_install
call :log "Starting fresh installation"
:: Prompt for GPU installation
set /p gpu_support="Do you want to install with GPU support? (y/n): "
if /i "%gpu_support%"=="y" (
    set /p gpu_type="Choose GPU type (1 for CUDA, 2 for AMD): "
    if "!gpu_type!"=="1" (
        echo Please ensure your NVIDIA GPU drivers and CUDA are up to date.
        echo Visit https://developer.nvidia.com/cuda-downloads for instructions.
        set "gpu_choice=cuda"
    ) else if "!gpu_type!"=="2" (
        echo Please ensure your AMD GPU drivers are up to date.
        set "gpu_choice=amd"
    ) else (
        echo Invalid choice. Defaulting to CPU installation.
        set "gpu_choice=cpu"
    )
) else (
    set "gpu_choice=cpu"
)

:: Save GPU choice
echo !gpu_choice! > "%install_dir%\gpu_choice.txt"

:: Clone the repository
git clone https://github.com/rmusser01/tldw "%install_dir%"
cd "%install_dir%"

call :setup_environment
goto :eof

:setup_environment
call :log "Setting up environment"
:: Create and activate virtual environment (if it doesn't exist)
if not exist "venv" (
    python -m venv .\venv
)
call .\venv\Scripts\activate.bat

:: Upgrade pip and install wheel
python -m pip install --upgrade pip wheel

:: Install PyTorch based on GPU support choice
if "!gpu_choice!"=="cuda" (
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
) else if "!gpu_choice!"=="amd" (
    pip install torch-directml
) else (
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    :: Update config.txt to use CPU
    powershell -Command "(gc config.txt) -replace 'cuda', 'cpu' | Out-File -encoding ASCII config.txt"
)

:: Install other requirements
pip install -r requirements.txt

:: Install ffmpeg (if not already installed)
if not exist "ffmpeg.exe" (
    call :install_ffmpeg
)
goto :eof

:install_ffmpeg
call :log "Installing ffmpeg"
echo Installing ffmpeg...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile 'ffmpeg.zip'"
if %errorlevel% neq 0 (
    call :log "Failed to download ffmpeg"
    echo Error: Failed to download ffmpeg. Please check your internet connection and try again.
    exit /b 1
)
powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg' -Force"
move ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe .
rmdir /s /q ffmpeg
del ffmpeg.zip
mkdir .\Bin
move ffmpeg.exe .\Bin
goto :eof

:cleanup
call :log "Performing cleanup"
:: Deactivate virtual environment

call .\venv\Scripts\deactivate.bat
goto :eof

:log
echo %date% %time% - %~1 >> "%log_file%"
goto :eof
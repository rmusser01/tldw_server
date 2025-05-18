@echo off
setlocal enabledelayedexpansion

:: TLDW Windows Installer and Updater Script

:: Base directory where this script is located
set "script_base_dir=%~dp0"
set "log_file=%script_base_dir%tldw_install_log.txt"

:: Repository and target application directory names
set "repo_name=tldw"
set "app_subdir=PoC_Version"

:: Full path to the cloned repository and the application directory
set "install_dir=%script_base_dir%%repo_name%"
set "app_install_dir=%install_dir%\%app_subdir%"

:: --- Logging ---
:log
echo %date% %time% - %~1 >> "%log_file%"
goto :eof

call :log "--- Starting TLDW installation/update process ---"

:: --- Prerequisite Checks ---
call :log "Checking for Python..."
python --version >nul 2>&1
if %errorlevel% neq 0 (
    call :log "Python is not installed or not in PATH."
    echo Python is not installed or not in PATH.
    echo Please download and install Python from https://www.python.org/downloads/windows/
    echo Ensure Python and Pip are added to your system's PATH during installation.
    echo After installation, run this script again.
    pause
    exit /b 1
)
call :log "Python found."

call :log "Checking for Git..."
git --version >nul 2>&1
if %errorlevel% neq 0 (
    call :log "Git is not installed or not in PATH."
    echo Git is not installed or not in PATH.
    echo Please download and install Git from https://git-scm.com/download/win
    echo Ensure Git is added to your system's PATH during installation.
    echo After installation, run this script again.
    pause
    exit /b 1
)
call :log "Git found."

:: --- Main Logic ---
if exist "%install_dir%\.git" (
    set /p update_choice="TLDW repository found in '%install_dir%'. Do you want to update it? (y/n) [y]: "
    if /i "!update_choice!"=="" set "update_choice=y"
    if /i "!update_choice!"=="y" (
        call :update_installation
    ) else (
        call :log "User chose not to update existing installation."
        echo Skipping update. To do a fresh install, remove the '%install_dir%' directory and re-run.
        if not exist "%app_install_dir%\venv" (
            echo Virtual environment not found. Setting it up...
            call :setup_environment
        )
    )
) else (
    call :fresh_installation
)

call :log "--- TLDW installation/update process completed ---"
echo.
echo Installation/Update completed!

set /p run_now_choice="Do you want to run TLDW now? (y/n) [y]: "
if /i "!run_now_choice!"=="" set "run_now_choice=y"
if /i "!run_now_choice!"=="y" (
    echo Attempting to run TLDW...
    cd /D "%app_install_dir%"
    if errorlevel 1 (
        call :log "Failed to cd into %app_install_dir% to run script."
        echo Error: Could not navigate to %app_install_dir% to run.
        pause
        exit /b 1
    )

    if not exist "summarize.py" (
        call :log "summarize.py not found in %app_install_dir%."
        echo Error: summarize.py not found. The application structure might be incorrect.
        pause
        exit /b 1
    )

    call :log "Activating venv to run summarize.py"
    if exist ".\venv\Scripts\activate.bat" (
        call .\venv\Scripts\activate.bat
    ) else (
        call :log "venv\Scripts\activate.bat not found."
        echo Error: Could not find venv activation script.
        pause
        exit /b 1
    )

    call :log "Running: python summarize.py -gui -log DEBUG"
    echo Starting TLDW GUI... (Press Ctrl+C in the new window to exit)
    start "TLDW" cmd /c "python summarize.py -gui -log DEBUG & pause"

    call :log "Deactivating venv (this script's context)."
    if exist ".\venv\Scripts\deactivate.bat" (
       call .\venv\Scripts\deactivate.bat
    )
    echo TLDW application started in a new window.
) else (
    echo You can run TLDW later by navigating to "%app_install_dir%" and running ".\venv\Scripts\activate.bat" then "python summarize.py -gui".
)

echo Script finished. Check "%log_file%" for details.
pause
endlocal
exit /b 0


:: --- Update Function ---
:update_installation
call :log "Starting update for existing installation in %install_dir%"
cd /D "%install_dir%"
if errorlevel 1 (
    call :log "Failed to cd into %install_dir%"
    echo Error: Could not navigate to %install_dir%
    exit /b 1
)

call :log "Fetching remote changes..."
git fetch
if %errorlevel% neq 0 (
    call :log "git fetch failed."
    echo Error: git fetch failed. Check your internet connection or git configuration.
    exit /b 1
)

for /f %%i in ('git rev-parse HEAD') do set "old_version=%%i"
for /f %%i in ('git rev-parse @{u}') do set "new_version=%%i"

if "!old_version!"=="!new_version!" (
    echo TLDW is already up to date (Version: !old_version!).
    call :log "TLDW is already up to date."
    if not exist "%app_install_dir%\venv" (
        echo Virtual environment not found. Setting it up...
        call :setup_environment
    ) else (
        echo To re-initialize the environment, consider removing the 'venv' directory and re-running the installer.
    )
    goto :eof
)

echo Current version: !old_version!
echo New version available: !new_version!
set /p confirm_update="Do you want to proceed with the update? (y/n): "
if /i "!confirm_update!"=="y" (
    call :log "User confirmed update. Creating backup..."
    set "timestamp=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
    set "timestamp=!timestamp::=_!"
    set "timestamp=!timestamp:/=_!"
    set "backup_name=%repo_name%_backup_!timestamp!"
    xcopy /E /I /H /Y "%install_dir%" "%script_base_dir%%backup_name%\"
    if %errorlevel% equ 0 (
        call :log "Backup created at %script_base_dir%%backup_name%"
    ) else (
        call :log "Failed to create backup. Error code: %errorlevel%. Continuing without backup."
        echo Warning: Failed to create backup.
    )

    call :log "Pulling latest changes..."
    git pull
    if %errorlevel% neq 0 (
        call :log "git pull failed."
        echo Error: git pull failed. Please check for conflicts or stash local changes.
        exit /b 1
    )
    call :log "Successfully pulled latest changes."
    echo TLDW updated successfully.
    call :setup_environment
) else (
    call :log "Update cancelled by user."
    echo Update cancelled.
)
goto :eof


:: --- Fresh Install Function ---
:fresh_installation
call :log "Starting fresh installation into %install_dir%"

if exist "%install_dir%" (
    set /p remove_confirm="Directory '%install_dir%' already exists. Do you want to remove it and proceed with a fresh install? (y/n): "
    if /i "!remove_confirm!"=="y" (
        call :log "Removing existing directory: %install_dir%"
        rmdir /S /Q "%install_dir%"
        if errorlevel 1 (
            call :log "Failed to remove existing directory %install_dir%"
            echo Error: Failed to remove existing directory. Please remove it manually.
            exit /b 1
        )
    ) else (
        call :log "Fresh installation aborted by user due to existing directory."
        echo Installation aborted. Directory '%install_dir%' was not removed.
        exit /b 0
    )
)

mkdir "%install_dir%" 2>nul

set "gpu_choice=cpu"
set /p gpu_support="Do you want to install with GPU support? (y/n) [n]: "
if /i "!gpu_support!"=="" set "gpu_support=n"

if /i "!gpu_support!"=="y" (
    echo Select GPU type:
    echo   1) NVIDIA (CUDA)
    echo   2) AMD (DirectML)
    echo   *) CPU (default)
    set /p gpu_type_choice="Enter choice [CPU]: "
    if /i "!gpu_type_choice!"=="" set "gpu_type_choice=CPU"

    if "!gpu_type_choice!"=="1" (
        echo Configuring for NVIDIA CUDA support.
        echo Please ensure your NVIDIA drivers and CUDA Toolkit (version compatible with PyTorch) are installed.
        echo Refer to: https://developer.nvidia.com/cuda-downloads and https://pytorch.org/get-started/locally/
        set "gpu_choice=cuda"
    ) else if "!gpu_type_choice!"=="2" (
        echo Configuring for AMD GPU support (DirectML for Windows).
        echo Please ensure your AMD drivers are up to date.
        echo Refer to: https://pytorch.org/get-started/locally/ (select DirectML)
        set "gpu_choice=amd"
    ) else (
        echo Invalid choice or no choice. Defaulting to CPU installation.
        set "gpu_choice=cpu"
    )
) else (
    echo Proceeding with CPU-only installation.
    set "gpu_choice=cpu"
)
call :log "GPU choice set to: !gpu_choice!"

call :log "Cloning repository https://github.com/rmusser01/tldw into %install_dir%"
git clone https://github.com/rmusser01/tldw "%install_dir%"
if %errorlevel% neq 0 (
    call :log "git clone failed."
    echo Error: Could not clone the repository. Check your internet connection and git installation.
    exit /b 1
)
call :log "Repository cloned successfully."

call :setup_environment
goto :eof


:: --- Environment Setup Function ---
:setup_environment
cd /D "%app_install_dir%"
if errorlevel 1 (
    call :log "Failed to cd into %app_install_dir% for environment setup."
    echo Error: Could not navigate to %app_install_dir%
    exit /b 1
)
call :log "Setting up Python virtual environment in %app_install_dir%\venv"
if not exist "venv" (
    python -m venv .\venv
    if %errorlevel% neq 0 (
        call :log "Failed to create virtual environment."
        echo Error: Failed to create Python virtual environment.
        exit /b 1
    )
    call :log "Virtual environment created."
) else (
    call :log "Virtual environment already exists."
)

call :log "Activating virtual environment."
call .\venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    call :log "Failed to activate virtual environment."
    echo Error: Failed to activate Python virtual environment.
    exit /b 1
)

call :log "Upgrading pip and installing wheel."
python -m pip install --upgrade pip wheel
if %errorlevel% neq 0 (
    call :log "Failed to upgrade pip or install wheel."
    echo Error: Failed to upgrade pip or install wheel.
    call .\venv\Scripts\deactivate.bat
    exit /b 1
)

:: Write/Update gpu_choice.txt inside PoC_Version
echo !gpu_choice! > "%app_install_dir%\gpu_choice.txt"
call :log "Saved/Updated gpu_choice.txt with: !gpu_choice!"


:: Ensure Config_Files directory and config.txt exist
if not exist "Config_Files" mkdir "Config_Files"
if not exist "Config_Files\config.txt" (
    call :log "Config_Files\config.txt not found. Creating a default."
    (
        echo [Processing]
        echo processing_choice = !gpu_choice!
        echo [Database]
        echo sqlite_path = Databases/media_summary.db
        echo rag_qa_db_path = Databases/rag_qa.db
        echo chatDB_path = Databases/chatDB.db
        echo prompts_db_path = Databases/prompts.db
        echo chroma_db_path = Databases/chroma_db
        :: Add other essential default config sections/keys if needed
    ) > "Config_Files\config.txt"
)


call :log "Installing PyTorch (!gpu_choice! version)..."
if /i "!gpu_choice!"=="cuda" (
    python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    if %errorlevel% neq 0 (
        call :log "Failed to install PyTorch with CUDA. Check CUDA/driver compatibility or network. Attempting CPU fallback."
        echo Error: Failed to install PyTorch with CUDA. Attempting CPU version...
        python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
        if %errorlevel% neq 0 (
            call :log "Failed to install PyTorch CPU version as fallback."
            echo Error: Failed to install PyTorch CPU version.
            call .\venv\Scripts\deactivate.bat
            exit /b 1
        )
        call :log "PyTorch CPU version installed as fallback."
        powershell -Command "(gc Config_Files\config.txt) -replace 'processing_choice = cuda', 'processing_choice = cpu' | Set-Content -Path Config_Files\config.txt -Encoding ascii"
        call :log "Updated config.txt to use CPU due to PyTorch CUDA installation failure."
    ) else (
        call :log "PyTorch with CUDA installed successfully."
        powershell -Command "(gc Config_Files\config.txt) -replace 'processing_choice = cpu', 'processing_choice = cuda' | Set-Content -Path Config_Files\config.txt -Encoding ascii"
        call :log "Ensured config.txt is set to CUDA."
    )
) else if /i "!gpu_choice!"=="amd" (
    python -m pip install torch-directml
    if %errorlevel% neq 0 (
        call :log "Failed to install torch-directml for AMD GPU. Attempting CPU fallback."
        echo Error: Failed to install torch-directml. Attempting CPU version...
        python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
        if %errorlevel% neq 0 (
             call :log "Failed to install PyTorch CPU version as fallback."
             echo Error: Failed to install PyTorch CPU version.
             call .\venv\Scripts\deactivate.bat
             exit /b 1
        )
        call :log "PyTorch CPU version installed as fallback."
        powershell -Command "(gc Config_Files\config.txt) -replace 'processing_choice = .+', 'processing_choice = cpu' | Set-Content -Path Config_Files\config.txt -Encoding ascii"
        call :log "Updated config.txt to use CPU due to torch-directml installation failure."
    ) else (
        call :log "torch-directml for AMD GPU installed successfully."
        powershell -Command "(gc Config_Files\config.txt) -replace 'processing_choice = cpu', 'processing_choice = directml' | Set-Content -Path Config_Files\config.txt -Encoding ascii"
        call :log "Ensured config.txt is set to DirectML."
    )
) else (
    python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    if %errorlevel% neq 0 (
         call :log "Failed to install PyTorch CPU version."
         echo Error: Failed to install PyTorch CPU version.
         call .\venv\Scripts\deactivate.bat
         exit /b 1
    )
    call :log "PyTorch CPU version installed successfully."
    powershell -Command "(gc Config_Files\config.txt) -replace 'processing_choice = .+', 'processing_choice = cpu' | Set-Content -Path Config_Files\config.txt -Encoding ascii"
    call :log "Updated config.txt to use CPU."
)

call :log "Installing other requirements from requirements.txt"
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        call :log "Failed to install packages from requirements.txt"
        echo Error: Failed to install required packages from requirements.txt. Check the file and your network.
        call .\venv\Scripts\deactivate.bat
        exit /b 1
    )
    call :log "Successfully installed packages from requirements.txt"
) else (
    call :log "requirements.txt not found in %app_install_dir%. Skipping."
    echo Warning: requirements.txt not found. Some features might not work.
)

:: Install ffmpeg (if not already in Bin directory)
if not exist "Bin\ffmpeg.exe" (
    call :install_ffmpeg
) else (
    call :log "ffmpeg.exe already exists in Bin directory."
)

call :log "Environment setup complete."
goto :eof


:: --- FFmpeg Install Function ---
:install_ffmpeg
call :log "Attempting to install ffmpeg..."
echo Installing ffmpeg (this might take a moment)...

:: Create Bin directory if it doesn't exist
if not exist "Bin" mkdir "Bin"

:: Download using PowerShell for better reliability
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    $ErrorActionPreference = 'Stop'; ^
    $ProgressPreference = 'SilentlyContinue'; ^
    try { ^
        Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg-essentials.zip'; ^
        Write-Host 'ffmpeg downloaded.'; ^
    } catch { ^
        Write-Error ('Failed to download ffmpeg: {0}' -f $_.Exception.Message); ^
        exit 1; ^
    }

if not exist "ffmpeg-essentials.zip" (
    call :log "ffmpeg download failed (ffmpeg-essentials.zip not found)."
    echo Error: Failed to download ffmpeg.zip.
    goto :eof
)

call :log "Extracting ffmpeg..."
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    $ErrorActionPreference = 'Stop'; ^
    try { ^
        Expand-Archive -Path 'ffmpeg-essentials.zip' -DestinationPath 'ffmpeg_temp' -Force; ^
        Write-Host 'ffmpeg extracted.'; ^
    } catch { ^
        Write-Error ('Failed to extract ffmpeg.zip: {0}' -f $_.Exception.Message); ^
        exit 1; ^
    }

if not exist "ffmpeg_temp" (
    call :log "ffmpeg extraction failed (ffmpeg_temp directory not found)."
    echo Error: Failed to extract ffmpeg.zip.
    if exist "ffmpeg-essentials.zip" del "ffmpeg-essentials.zip"
    goto :eof
)

:: Find ffmpeg.exe in the extracted structure. It's usually in a 'bin' subfolder of a versioned folder.
:: Example: ffmpeg_temp\ffmpeg-6.0-essentials_build\bin\ffmpeg.exe
set "ffmpeg_exe_path="
for /R "ffmpeg_temp" %%F in (ffmpeg.exe) do (
    set "ffmpeg_exe_path=%%F"
    goto found_ffmpeg
)

:found_ffmpeg
if defined ffmpeg_exe_path (
    if exist "!ffmpeg_exe_path!" (
        call :log "Found ffmpeg.exe at !ffmpeg_exe_path!"
        move /Y "!ffmpeg_exe_path!" "Bin\ffmpeg.exe"
        if %errorlevel% equ 0 (
            call :log "ffmpeg.exe moved to Bin directory."
        ) else (
            call :log "Failed to move ffmpeg.exe to Bin. Error: %errorlevel%"
            echo Error: Failed to move ffmpeg.exe.
        )
    ) else (
        call :log "ffmpeg.exe found by search but path seems invalid: !ffmpeg_exe_path!"
        echo Error: Could not locate ffmpeg.exe after extraction.
    )
) else (
    call :log "ffmpeg.exe not found within extracted files."
    echo Error: ffmpeg.exe not found after extraction.
)

:: Cleanup
if exist "ffmpeg_temp" rmdir /S /Q "ffmpeg_temp"
if exist "ffmpeg-essentials.zip" del "ffmpeg-essentials.zip"
call :log "ffmpeg installation attempt finished."
goto :eof
@echo off
setlocal enabledelayedexpansion

:: TLDW Run Script

set "install_dir=%~dp0tldw"

if not exist "%install_dir%" (
    echo TLDW installation not found. Please run the install_update_tldw.bat script first.
    pause
    exit /b 1
)

cd "%install_dir%"

:: Activate virtual environment
call .\venv\Scripts\activate.bat

:: Run TLDW
python summarize.py -gui

:: Deactivate virtual environment when done
call .\venv\Scripts\deactivate.bat

exit /b 0
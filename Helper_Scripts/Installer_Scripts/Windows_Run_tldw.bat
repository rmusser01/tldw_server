@echo off
setlocal

:: Compatibility launcher for installs created by Windows_Install_Update.bat.

set "install_dir=%~dp0tldw"
if not "%TLDW_INSTALL_DIR%"=="" set "install_dir=%TLDW_INSTALL_DIR%"
set "launcher=%install_dir%\quick-launch.ps1"

if not exist "%launcher%" (
    echo tldw_server launcher not found at: %launcher%
    echo Run Windows_Install_Update.bat first, or set TLDW_INSTALL_DIR to a checkout that contains quick-launch.ps1.
    exit /b 1
)

if "%TLDW_VENV_DIR%"=="" set "TLDW_VENV_DIR=venv"
if "%TLDW_SKIP_INSTALL%"=="" set "TLDW_SKIP_INSTALL=1"

powershell -ExecutionPolicy Bypass -File "%launcher%" %*
exit /b %errorlevel%

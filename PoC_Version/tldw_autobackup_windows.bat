@echo off

REM Path to your Litestream configuration
set LITESTREAM_CONFIG=litestream.yml

REM Command to start your application
set APP_COMMAND=python summarize.py -gui

REM Log file path
set LOG_FILE=application.log

REM Start Litestream with restore and replication, logging output
litestream replicate ^
  -config "%LITESTREAM_CONFIG%" ^
  -exec "%APP_COMMAND%" ^
  -restore -if-replica-exists ^
  -v 2>&1 | powershell -Command "Foreach ($line in [Console]::In.ReadToEnd().Split(\"`n\")) { $timestamp = Get-Date -Format '[yyyy-MM-dd HH:mm:ss]'; $output = \"$timestamp $line\"; Write-Host $output; Add-Content '%LOG_FILE%' $output }"

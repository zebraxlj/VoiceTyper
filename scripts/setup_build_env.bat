@echo off
setlocal

cd /d "%~dp0.."

echo INFO - Checking uv...
uv --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR - uv is not installed or not on PATH.
    goto :end
)

echo INFO - Syncing project dependencies...
uv sync
if %ERRORLEVEL% neq 0 (
    echo ERROR - Failed to sync dependencies.
    goto :end
)
echo INFO - Build environment ready.

:end
endlocal
pause

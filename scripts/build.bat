@echo off
setlocal

cd /d "%~dp0.."

:: Setup build environment
echo INFO - Checking uv...
uv --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR - uv is not installed or not on PATH.
    goto :fail
)

echo INFO - Syncing project dependencies...
uv sync
if %ERRORLEVEL% neq 0 (
    echo ERROR - Failed to sync dependencies.
    goto :fail
)
echo INFO - Build environment ready.

:: Clean previous build artifacts
echo INFO - Cleaning previous build...
if exist build rmdir /s /q build
if exist VoiceTyper.spec del /q VoiceTyper.spec
if exist dist\VoiceTyper.exe del /q dist\VoiceTyper.exe

:: Build
echo INFO - Building VoiceTyper.exe...
uv run --with pyinstaller pyinstaller ^
    --onefile ^
    --noconsole ^
    --name "VoiceTyper" ^
    --distpath dist ^
    --workpath build ^
    --specpath . ^
    --collect-all sherpa_onnx ^
    --collect-all pypinyin ^
    --hidden-import "pynput.keyboard._win32" ^
    --hidden-import "pynput.mouse._win32" ^
    --hidden-import "pystray._win32" ^
    --hidden-import "PIL.Image" ^
    --hidden-import "PIL.ImageDraw" ^
    examples\demo_push_to_talk_ui.py

if %ERRORLEVEL% neq 0 goto :fail

echo.
echo INFO - Build succeeded: dist\VoiceTyper.exe
explorer dist
goto :end

:fail
echo.
echo ERROR - Build FAILED.

:end
endlocal
pause

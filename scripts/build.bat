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

echo INFO - Ensuring managed Python is available...
uv python install
if %ERRORLEVEL% neq 0 (
    echo ERROR - Failed to install managed Python.
    goto :fail
)

echo INFO - Syncing project dependencies...
uv sync --python-preference only-managed
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
uv run --python-preference only-managed --with pyinstaller pyinstaller ^
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
    --icon "UI\assets\IconApp.ico" ^
    --add-data "UI\assets\IconApp.png;UI\assets" ^
    --add-data "UI\assets\IconTaskTray.png;UI\assets" ^
    examples\demo_push_to_talk_ui.py

if %ERRORLEVEL% neq 0 goto :fail

echo.
echo INFO - Build succeeded: dist\VoiceTyper.exe
powershell -NoProfile -Command ^
    "$f=(Resolve-Path dist).Path; $n='VoiceTyper.exe';" ^
    "$s=New-Object -Com Shell.Application;" ^
    "$w=$s.Windows()|Where-Object{$_.Document.Folder.Self.Path -eq $f}|Select-Object -First 1;" ^
    "if($w){" ^
    "  Add-Type -Name U -Namespace Win32 -MemberDefinition '[DllImport(\"user32.dll\")]public static extern bool SetForegroundWindow(IntPtr h);';" ^
    "  [Win32.U]::SetForegroundWindow([IntPtr]$w.HWND)|Out-Null;" ^
    "  $i=$w.Document.Folder.ParseName($n);" ^
    "  $w.Document.SelectItem($i,29)" ^
    "}else{" ^
    "  explorer /select,(Join-Path $f $n)" ^
    "}"
goto :end

:fail
echo.
echo ERROR - Build FAILED.

:end
endlocal
pause

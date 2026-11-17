@echo off
REM ==========================================================
REM  HD-MEA Analysis - starter
REM  Uses the "hdmea-gui" conda env made by install.bat.
REM  To use your existing notebook environment instead, change
REM  ENV_NAME below (it then only needs:  pip install PySide6).
REM  The console window shows SpikeInterface / HerdingSpikes progress.
REM ==========================================================
setlocal
cd /d "%~dp0"
set ENV_NAME=hdmea-gui
REM env_name.txt (one line: env name or full path) overrides this and is never shipped in updates
if exist "%~dp0env_name.txt" set /p ENV_NAME=<"%~dp0env_name.txt"

where conda >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
        call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate base
    ) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate base
    ) else if exist "%USERPROFILE%\Anaconda\condabin\conda.bat" (
        call "%USERPROFILE%\Anaconda\condabin\conda.bat" activate base
    )
)
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo Could not activate conda env %ENV_NAME%. Run install.bat first.
    pause
    exit /b 1
)
python -m hdmea %*
if errorlevel 1 pause

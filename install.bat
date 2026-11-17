@echo off
REM ==========================================================
REM  HD-MEA Analysis - installer (Windows, Miniconda/Anaconda)
REM  Creates a conda environment "hdmea-gui" and installs the
REM  dependencies. Run once. Re-running updates the packages.
REM ==========================================================
setlocal
cd /d "%~dp0"
set ENV_NAME=hdmea-gui

where conda >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
        call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate base
    ) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate base
    ) else if exist "%USERPROFILE%\Anaconda\condabin\conda.bat" (
        call "%USERPROFILE%\Anaconda\condabin\conda.bat" activate base
    ) else (
        echo Conda was not found. Open this file from the "Anaconda Prompt" or install Miniconda.
        pause
        exit /b 1
    )
)

call conda env list | findstr /B /C:"%ENV_NAME% " >nul
if errorlevel 1 (
    echo Creating conda environment %ENV_NAME% ...
    call conda create -y -n %ENV_NAME% python=3.11
    if errorlevel 1 goto :fail
)
call conda activate %ENV_NAME%
if errorlevel 1 goto :fail

python -m pip install --upgrade pip
REM core packages first (pip is all-or-nothing, so HerdingSpikes is installed separately)
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail
python -c "import matplotlib, PySide6, spikeinterface, zarr, h5py, skimage" || goto :fail

echo.
echo Installing HerdingSpikes (optional - needed only to run sorting from the GUI)...
python -m pip install herdingspikes
if errorlevel 1 (
    echo.
    echo HerdingSpikes could not be installed here. The GUI works without it for everything
    echo except "Run HerdingSpikes". Try:  conda install -c conda-forge herdingspikes
    echo or sort in your notebook environment and load the hs_full folder in the Sorting tab.
)

echo.
echo Installed. Start the program with start.bat
pause
exit /b 0

:fail
echo Installation failed.
pause
exit /b 1

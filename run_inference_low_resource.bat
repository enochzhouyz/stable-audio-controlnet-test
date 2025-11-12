@echo off
REM Batch script to run low-resource inference on Windows
REM For RTX 3050 (4GB VRAM) and similar GPUs

echo ========================================
echo Stable Audio ControlNet - Low Resource Inference
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Set default parameters (you can modify these)
set CHECKPOINT=
set INPUT_AUDIO=res\track_musdb\input.wav
set OUTPUT_DIR=outputs
set DEPTH_FACTOR=0.1
set STEPS=25
set CFG_SCALE=5.0
set MAX_DURATION=10.0

echo Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   Input Audio: %INPUT_AUDIO%
echo   Output Directory: %OUTPUT_DIR%
echo   Depth Factor: %DEPTH_FACTOR%
echo   Steps: %STEPS%
echo   CFG Scale: %CFG_SCALE%
echo   Max Duration: %MAX_DURATION%s
echo.

REM Check if input audio exists
if not exist "%INPUT_AUDIO%" (
    echo WARNING: Input audio file not found: %INPUT_AUDIO%
    echo Please specify a valid audio file path.
    echo.
    echo Usage: Edit this batch file and set INPUT_AUDIO variable
    pause
    exit /b 1
)

echo Starting inference...
echo This may take several minutes on 4GB VRAM...
echo.

REM Build command
set CMD=python inference_low_resource.py --input_audio "%INPUT_AUDIO%" --output_dir "%OUTPUT_DIR%" --depth_factor %DEPTH_FACTOR% --steps %STEPS% --cfg_scale %CFG_SCALE% --max_duration %MAX_DURATION%

REM Add checkpoint if specified
if not "%CHECKPOINT%"=="" (
    set CMD=%CMD% --checkpoint "%CHECKPOINT%"
)

REM Run the command
%CMD%

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Inference failed!
    echo ========================================
    echo.
    echo Common solutions:
    echo   1. Out of memory: Reduce DEPTH_FACTOR to 0.05
    echo   2. Out of memory: Reduce STEPS to 10
    echo   3. Out of memory: Reduce MAX_DURATION to 5
    echo   4. Missing dependencies: pip install -r requirements.txt
    echo   5. CUDA error: Update NVIDIA drivers
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Inference completed successfully!
echo ========================================
echo Output saved to: %OUTPUT_DIR%
echo.
pause


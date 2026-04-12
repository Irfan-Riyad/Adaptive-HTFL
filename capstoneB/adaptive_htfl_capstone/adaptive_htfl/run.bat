@echo off
echo Starting Adaptive-HTFL Project...

cd /d a:\capstoneB\adaptive_htfl_capstone\adaptive_htfl

echo Step 1: Running federated learning experiment...
a:\capstoneB\.venv\Scripts\python.exe run_experiment.py
if %errorlevel% neq 0 (
    echo Experiment failed!
    pause
    exit /b 1
)

echo Step 2: Generating dashboard figures...
a:\capstoneB\.venv\Scripts\python.exe dashboard.py
if %errorlevel% neq 0 (
    echo Dashboard generation failed!
    pause
    exit /b 1
)

echo Step 3: Starting web server...
cd results\figures
echo Dashboard ready! Open http://localhost:8000/index.html in your browser
a:\capstoneB\.venv\Scripts\python.exe -m http.server 8000

pause

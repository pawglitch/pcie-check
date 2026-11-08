@echo off
rem pcie-check launcher for Windows - double-click to run.
cd /d "%~dp0"
set GPU=0
set MIN=10
py pcie-check.py --list
echo.
set /p GPU=GPU index, or "all" [0]: 
set /p MIN=Minutes [10]: 
py pcie-check.py %GPU% %MIN% --json pcie-check-result.json
echo.
pause

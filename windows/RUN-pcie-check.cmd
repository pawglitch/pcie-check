@echo off
cd /d "%~dp0"
set GPU=0
set MIN=10
set /p GPU=GPU index [0]: 
set /p MIN=Minutes [10]: 
py "%~dp0pcie-check.py" %GPU% %MIN%
echo.
pause

@echo off
title InsightOptima
cd /d C:\Projects\InsightOptima

echo.
echo  InsightOptima - Starting dashboard...
echo  Project path: C:\Projects\InsightOptima
echo.

REM Stop any old Streamlit still running on port 8501 (from previous location)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo  Stopping old process on port 8501 (PID %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

echo  After startup, open: http://localhost:8501
echo  Keep this window open while using the app.
echo.

python run_app.py

echo.
pause

@echo off
title InsightOptima - Stop
echo Stopping any InsightOptima / Streamlit on port 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo Killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)
echo Done. Port 8501 should be free now.
timeout /t 2 >nul

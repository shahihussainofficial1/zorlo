@echo off
rem Zorlo: double-click to start. Shows the AI models you can use, downloads the one you pick
rem (the first time), keeps it up to date, and opens the chat page in your browser.
title Zorlo
cd /d "%~dp0"
"%~dp0zorlo.exe" start
echo.
pause
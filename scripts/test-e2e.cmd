@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test-e2e.ps1" %*

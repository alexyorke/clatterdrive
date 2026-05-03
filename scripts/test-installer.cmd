@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test-installer.ps1" %*

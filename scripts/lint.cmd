@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lint.ps1" %*

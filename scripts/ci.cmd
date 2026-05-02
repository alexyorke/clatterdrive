@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ci.ps1" %*

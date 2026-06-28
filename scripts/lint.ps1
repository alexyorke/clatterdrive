$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "Use-RepoUv.ps1")
Enable-RepoUvFallbacks

Invoke-Uv run python -m compileall -q clatterdrive tools tests main.py smoke.py
Invoke-Uv run ruff check .
Invoke-Uv run python -m vulture clatterdrive tools tests main.py smoke.py --min-confidence 80
Invoke-Uv run python -m mypy .

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

uv run python -m compileall -q clatterdrive tools tests main.py smoke.py
uv run ruff check .
uv run python -m vulture clatterdrive tools tests main.py smoke.py --min-confidence 80
uv run python -m mypy .

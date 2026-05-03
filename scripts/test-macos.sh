#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS tests must run on macOS." >&2
  exit 1
fi

bash "${script_dir}/bootstrap-macos.sh"

export PYTHONPATH="${repo_root}"
pytest_workers="${CLATTERDRIVE_MACOS_PYTEST_WORKERS:-2}"
uv run pytest -n "${pytest_workers}"
uv run ruff check clatterdrive tests tools
uv run mypy clatterdrive tests tools
swift test --package-path launcher/ClatterDrive.MacLauncher

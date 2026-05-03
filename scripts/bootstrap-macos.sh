#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS bootstrap must run on macOS." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "Swift toolchain not found. Install Xcode command line tools with: xcode-select --install" >&2
  exit 1
fi

uv sync --locked --group dev
python3 --version
uv --version
swift --version
command -v hdiutil >/dev/null
command -v codesign >/dev/null
command -v mount_webdav >/dev/null

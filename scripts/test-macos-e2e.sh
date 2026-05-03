#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS E2E tests must run on macOS." >&2
  exit 1
fi

packaged=0
from_dmg=0
include_mount=0
use_existing_package=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --packaged)
      packaged=1
      ;;
    --from-dmg)
      from_dmg=1
      packaged=1
      ;;
    --include-mount)
      include_mount=1
      ;;
    --use-existing-package)
      use_existing_package=1
      packaged=1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done

bash "${script_dir}/bootstrap-macos.sh"
export PYTHONPATH="${repo_root}"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/clatterdrive-macos-e2e.XXXXXX")"
mount_dir=""
cleanup() {
  if [[ -n "${mount_dir}" && -d "${mount_dir}" ]]; then
    hdiutil detach "${mount_dir}" -quiet || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

uv run clatterdrive profiles --json > "${tmp_dir}/profiles.json"
uv run clatterdrive doctor --json --audio off --port 0 --backing-dir "${tmp_dir}/doctor backing" > "${tmp_dir}/doctor.json"

backend_args=()
if [[ "${packaged}" -eq 1 ]]; then
  machine_arch="$(uname -m)"
  if [[ "${CLATTERDRIVE_MACOS_ARCH:-${machine_arch}}" == "arm64" || "${CLATTERDRIVE_MACOS_ARCH:-${machine_arch}}" == "ARM64" ]]; then
    artifact_arch="arm64"
  else
    artifact_arch="x64"
  fi
  if [[ "${use_existing_package}" -eq 0 ]]; then
    bash "${script_dir}/package-macos.sh"
  fi
  app_bundle="${repo_root}/.runtime/package/macos-${artifact_arch}/ClatterDrive.app"
  if [[ "${from_dmg}" -eq 1 ]]; then
    dmg_path="${repo_root}/.runtime/package/ClatterDrive-macos-${artifact_arch}.dmg"
    mount_dir="${tmp_dir}/dmg"
    mkdir -p "${mount_dir}"
    hdiutil attach "${dmg_path}" -mountpoint "${mount_dir}" -nobrowse -quiet
    app_bundle="${mount_dir}/ClatterDrive.app"
  fi
  backend_exe="${app_bundle}/Contents/Resources/backend/clatterdrive-backend"
  if [[ ! -x "${backend_exe}" ]]; then
    echo "Packaged backend not found or not executable: ${backend_exe}" >&2
    exit 1
  fi
  test -x "${app_bundle}/Contents/MacOS/ClatterDrive"
  test -f "${app_bundle}/Contents/Info.plist"
  backend_args=(--backend-exe "${backend_exe}")
fi

uv run python -m tools.backend_e2e "${backend_args[@]}"
uv run python -m tools.backend_e2e "${backend_args[@]}" --space-paths

if [[ "${include_mount}" -eq 1 ]]; then
  uv run python -m tools.backend_e2e "${backend_args[@]}" --macos-mount
fi

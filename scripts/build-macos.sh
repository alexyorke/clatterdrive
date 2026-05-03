#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS packaging must run on macOS because PyInstaller and SwiftUI artifacts are platform-specific." >&2
  exit 1
fi

bash "${script_dir}/bootstrap-macos.sh"

machine_arch="$(uname -m)"
case "${CLATTERDRIVE_MACOS_ARCH:-${machine_arch}}" in
  arm64|ARM64)
    artifact_arch="arm64"
    ;;
  x86_64|x64|X64)
    artifact_arch="x64"
    ;;
  *)
    echo "Unsupported macOS architecture: ${CLATTERDRIVE_MACOS_ARCH:-${machine_arch}}" >&2
    exit 1
    ;;
esac

runtime_dir="${repo_root}/.runtime"
pyinstaller_work="${runtime_dir}/build/pyinstaller-macos-${artifact_arch}"
dist_root="${runtime_dir}/dist/macos-${artifact_arch}"
backend_dist="${dist_root}/backend"
launcher_dist="${dist_root}/launcher"
mac_package="launcher/ClatterDrive.MacLauncher"

mkdir -p "${pyinstaller_work}" "${backend_dist}" "${launcher_dist}"

uv run pyinstaller \
  --noconfirm \
  --clean \
  --name clatterdrive-backend \
  --distpath "${backend_dist}" \
  --workpath "${pyinstaller_work}" \
  --collect-submodules clatterdrive \
  --collect-data wsgidav \
  --hidden-import wsgidav.error_printer \
  --hidden-import wsgidav.dir_browser._dir_browser \
  --hidden-import wsgidav.request_resolver \
  tools/clatterdrive_backend_entry.py

swift build --package-path "${mac_package}" -c release
swift_bin_dir="$(swift build --package-path "${mac_package}" -c release --show-bin-path)"
cp "${swift_bin_dir}/ClatterDriveMacLauncher" "${launcher_dist}/ClatterDrive"
chmod +x "${launcher_dist}/ClatterDrive"

echo "Backend: ${backend_dist}/clatterdrive-backend"
echo "Launcher: ${launcher_dist}/ClatterDrive"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS packaging must run on macOS." >&2
  exit 1
fi

bash "${script_dir}/build-macos.sh"

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

dist_root="${repo_root}/.runtime/dist/macos-${artifact_arch}"
backend_dir="${dist_root}/backend/clatterdrive-backend"
launcher_bin="${dist_root}/launcher/ClatterDrive"
staging_dir="${repo_root}/.runtime/package/macos-${artifact_arch}"
app_bundle="${staging_dir}/ClatterDrive.app"
dmg_path="${repo_root}/.runtime/package/ClatterDrive-macos-${artifact_arch}.dmg"
info_plist="${repo_root}/launcher/ClatterDrive.MacLauncher/Resources/Info.plist"

rm -rf "${staging_dir}"
mkdir -p \
  "${app_bundle}/Contents/MacOS" \
  "${app_bundle}/Contents/Resources/backend" \
  "${app_bundle}/Contents/Resources/docs" \
  "${app_bundle}/Contents/Resources/sample-backing"

cp "${launcher_bin}" "${app_bundle}/Contents/MacOS/ClatterDrive"
cp "${info_plist}" "${app_bundle}/Contents/Info.plist"
cp -R "${backend_dir}/." "${app_bundle}/Contents/Resources/backend/"
cp README.md "${app_bundle}/Contents/Resources/README.md"
cp -R docs/. "${app_bundle}/Contents/Resources/docs/"
chmod +x "${app_bundle}/Contents/MacOS/ClatterDrive"
chmod +x "${app_bundle}/Contents/Resources/backend/clatterdrive-backend"

if [[ -n "${APPLE_DEVELOPER_ID_APPLICATION:-}" ]]; then
  codesign --force --deep --options runtime --timestamp --sign "${APPLE_DEVELOPER_ID_APPLICATION}" "${app_bundle}"
else
  echo "No APPLE_DEVELOPER_ID_APPLICATION configured; packaging unsigned app."
fi

rm -f "${dmg_path}"
hdiutil create -volname "ClatterDrive" -srcfolder "${staging_dir}" -ov -format UDZO "${dmg_path}"

if [[ -n "${APPLE_DEVELOPER_ID_APPLICATION:-}" ]]; then
  codesign --force --timestamp --sign "${APPLE_DEVELOPER_ID_APPLICATION}" "${dmg_path}"
  codesign --verify --deep --strict --verbose=2 "${app_bundle}"
fi

if [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]]; then
  xcrun notarytool submit "${dmg_path}" \
    --apple-id "${APPLE_ID}" \
    --team-id "${APPLE_TEAM_ID}" \
    --password "${APPLE_APP_SPECIFIC_PASSWORD}" \
    --wait
  xcrun stapler staple "${dmg_path}"
else
  echo "Apple notarization secrets not configured; skipping notarization."
fi

hdiutil imageinfo "${dmg_path}" >/dev/null
echo "Package: ${dmg_path}"

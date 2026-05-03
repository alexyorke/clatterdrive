#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Tart macOS VM tests require an Apple Silicon Mac host." >&2
  exit 1
fi

vm_name="${CLATTERDRIVE_TART_VM:-clatterdrive-macos-runner}"
vm_image="${CLATTERDRIVE_TART_IMAGE:-ghcr.io/cirruslabs/macos-runner:tahoe}"
vm_user="${CLATTERDRIVE_TART_USER:-admin}"
vm_password="${CLATTERDRIVE_TART_PASSWORD:-admin}"
guest_repo="/Volumes/My Shared Files/clatterdrive"
runtime_dir="${repo_root}/.runtime/tart"
log_path="${runtime_dir}/${vm_name}.log"

mkdir -p "${runtime_dir}"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required to install Tart and sshpass." >&2
  exit 1
fi

if ! command -v tart >/dev/null 2>&1; then
  brew install cirruslabs/cli/tart
fi

if ! command -v sshpass >/dev/null 2>&1; then
  brew install cirruslabs/cli/sshpass
fi

if ! tart list | awk '{print $1}' | grep -Fxq "${vm_name}"; then
  tart clone "${vm_image}" "${vm_name}"
fi

if ! tart ip "${vm_name}" >/dev/null 2>&1; then
  nohup tart run --no-graphics --dir="clatterdrive:${repo_root}" "${vm_name}" >"${log_path}" 2>&1 &
fi

ip_address=""
for _ in $(seq 1 90); do
  ip_address="$(tart ip "${vm_name}" 2>/dev/null || true)"
  if [[ -n "${ip_address}" ]]; then
    if sshpass -p "${vm_password}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${vm_user}@${ip_address}" "test -d '${guest_repo}'" >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 5
done

if [[ -z "${ip_address}" ]]; then
  echo "Tart VM did not become reachable. See ${log_path}" >&2
  exit 1
fi

sshpass -p "${vm_password}" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${vm_user}@${ip_address}" "cat > /tmp/clatterdrive-local-vm-test.sh && bash /tmp/clatterdrive-local-vm-test.sh" <<'EOF'
set -euo pipefail

cd "/Volumes/My Shared Files/clatterdrive"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

bash scripts/test-macos.sh
bash scripts/test-macos-e2e.sh --include-mount
bash scripts/test-macos-e2e.sh --packaged --from-dmg --include-mount
EOF

# macOS VM and Runner Plan

Use GitHub-hosted macOS runners first for CI. For local VM testing, use Apple hardware. This repo now supports an Apple Silicon Tart path that mounts the working tree into a macOS VM and runs the same macOS scripts used by CI.

Do not try to create a local macOS VM on a Windows or Linux host. Apple's current macOS license terms are for Apple-branded systems and permit additional macOS virtual instances on Apple-branded computers you own or control for development/testing. That makes Hackintosh-style local VM setup the wrong path for this repo.

## Supported Paths

- Primary: GitHub-hosted macOS runners.
  - `macos-latest` for Apple Silicon arm64.
  - `macos-15-intel` for Intel x64.
  - CI runs the same `scripts/*-macos.sh` scripts used locally.
- Optional local Apple-hardware VM: UTM.
  - Use on Apple Silicon macOS 12+ for manual GUI validation.
  - Use UTM's macOS virtualization flow and Apple-provided restore image handling.
- Local automatable Apple-hardware VM: Tart.
  - Use on Apple Silicon macOS 13+ for CI-style VM images.
  - Best local choice for repeatable ClatterDrive build/package/E2E checks.
- Optional cloud fallback: AWS EC2 Mac Dedicated Host or another Mac cloud provider.
  - Use only if GitHub-hosted macOS runners are insufficient.
  - EC2 Mac requires a Dedicated Host and has a minimum allocation period, so it is not the default.

## Local Tart Runbook

On an Apple Silicon Mac with enough disk space for a macOS VM image:

```bash
bash scripts/test-macos-tart.sh
```

Defaults:

- VM image: `ghcr.io/cirruslabs/macos-runner:tahoe`
- VM name: `clatterdrive-macos-runner`
- VM credentials: `admin` / `admin`
- Host repo mount: `/Volumes/My Shared Files/clatterdrive`

Override defaults when needed:

```bash
CLATTERDRIVE_TART_IMAGE=ghcr.io/cirruslabs/macos-tahoe-xcode:latest \
CLATTERDRIVE_TART_VM=clatterdrive-tahoe-xcode \
bash scripts/test-macos-tart.sh
```

The script installs `tart` and `sshpass` with Homebrew if missing, clones the VM image if needed, starts the VM with the repo mounted, SSHes into the guest, installs `uv` if needed, then runs:

```bash
bash scripts/test-macos.sh
bash scripts/test-macos-e2e.sh --include-mount
bash scripts/test-macos-e2e.sh --packaged --from-dmg --include-mount
```

Stop the VM when done:

```bash
tart stop clatterdrive-macos-runner
```

## Manual UTM Runbook

Use this for interactive GUI checks of the SwiftUI launcher:

1. Install UTM on an Apple Silicon Mac.
2. Create a VM with `Virtualization -> macOS 12+`.
3. Let UTM download the compatible macOS IPSW, or provide an Apple restore image.
4. Enable a shared directory for this repo.
5. In the guest, mount the shared directory if needed:

```bash
mkdir -p ~/workspace
mount_virtiofs share ~/workspace
cd ~/workspace/fake_hdd_fuse
bash scripts/test-macos.sh
bash scripts/package-macos.sh
open .runtime/package/macos-arm64/ClatterDrive.app
```

## What Docker Can and Cannot Cover

Linux Docker is still useful for backend-only WebDAV/audio smoke tests. It cannot validate a macOS `.app`, Finder/WebDAV integration, Gatekeeper behavior, Developer ID signing, notarization, stapling, or native SwiftUI behavior.

Windows virtualization cannot legally or accurately replace macOS testing for this app.

## References

- [Apple macOS Tahoe Software License Agreement](https://www.apple.com/legal/sla/docs/macOSTahoe.pdf)
- [Apple Virtualization framework: running macOS in a VM on Apple Silicon](https://developer.apple.com/documentation/virtualization/running-macos-in-a-virtual-machine-on-apple-silicon)
- [GitHub-hosted macOS runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [UTM macOS guest docs](https://docs.getutm.app/guest-support/macos/)
- [Tart](https://github.com/cirruslabs/tart)
- [AWS EC2 Mac instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-mac-instances.html)

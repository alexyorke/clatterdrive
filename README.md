# ClatterDrive

> [!WARNING]
> `ClatterDrive` is a novelty project for entertainment purposes. Contributions are welcome, but this is not a serious or trustworthy HDD simulator, and it should not be treated as accurate for storage, acoustics, or performance work.

`ClatterDrive` makes a normal directory feel and sound more like a mechanical hard drive.

- It exposes a directory over WebDAV.
- It injects HDD-like latency into reads, writes, metadata churn, standby/wake, and write-back behavior.
- It synthesizes drive noise procedurally from the same runtime events.
- It does **not** use prerecorded HDD sound effects.

## Audio Demo

- [Listen in the browser](https://alexyorke.github.io/clatterdrive/)

Checked-in sample renders:

| Scenario | Why it sounds different | File |
| --- | --- | --- |
| Spin-up, idle, park | Desk-coupled startup/body with a final park transient | [spinup-idle-park.wav](samples/spinup-idle-park.wav) |
| Idle, standby, wake | Quieter state change, then wake-up and renewed activity | [idle-standby-wake.wav](samples/idle-standby-wake.wav) |
| Metadata storm | Brighter, busier short/medium seeks on a barer mounting profile | [metadata-storm.wav](samples/metadata-storm.wav) |

## Quick Start

This repo uses `uv` and the committed [uv.lock](uv.lock).

```powershell
uv sync --group dev
uv run python main.py
```

Or:

```powershell
uv run python -m clatterdrive
uv run clatterdrive
```

Default URL:

- `http://127.0.0.1:8080`

## Using It

The simulator serves `backing_storage/` by default. Use the WebDAV URL, not the backing directory directly, or you will bypass the latency/audio path.

Windows options:

- open `http://127.0.0.1:8080/` in Explorer as a network location
- upload/download with `curl.exe`
- map it as a WebDAV drive, for example `net use X: \\127.0.0.1@8080\DavWWWRoot /persistent:no`

Windows `curl.exe` example:

```powershell
$demoName = "demo-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
$uploadPath = Join-Path $env:TEMP "clatterdrive-upload.bin"
$downloadPath = Join-Path $env:TEMP "clatterdrive-download.bin"

"hello from clatterdrive" | Set-Content -Path $uploadPath -Encoding ascii

curl.exe -X MKCOL "http://127.0.0.1:8080/$demoName/"
curl.exe -T $uploadPath "http://127.0.0.1:8080/$demoName/file.bin"
curl.exe "http://127.0.0.1:8080/$demoName/file.bin" --output $downloadPath
```

Notes:

- `MKCOL` only works for a directory that does not already exist. If you reuse a name like `demo/`, `405 Method Not Allowed` is expected.
- The `C:\path\to\...` strings were placeholders; replace them with real paths, or use the temp-file example above as-is.
- `curl.exe --output` will fail if the parent directory does not exist.

WSL option:

- use `davfs2` and disable client-side locks for this server

Verified in WSL 2 (Ubuntu 22.04) on this machine:

```bash
sudo apt-get update
sudo apt-get install -y davfs2
mkdir -p ~/.davfs2 ~/mnt/clatterdrive
cat > ~/.davfs2/clatterdrive.conf <<'EOF'
use_locks 0
delay_upload 0
EOF
sudo mount -t davfs http://127.0.0.1:8080/ ~/mnt/clatterdrive -o uid=$(id -u),gid=$(id -g),file_mode=0644,dir_mode=0755,conf=$HOME/.davfs2/clatterdrive.conf
echo "hello from wsl davfs" > /tmp/clatterdrive-src.txt
cp /tmp/clatterdrive-src.txt ~/mnt/clatterdrive/wsl-copy-test.txt
cat ~/mnt/clatterdrive/wsl-copy-test.txt
cmp -s /tmp/clatterdrive-src.txt ~/mnt/clatterdrive/wsl-copy-test.txt && echo matches=true
```

WSL notes:

- `davfs2` will prompt for a username and password; press Enter for both because the local server is anonymous.
- `use_locks 0` is currently required for `davfs2` writes against this server. Without it, `davfs2` tries to lock a not-yet-created path and uploads fail with `Input/output error`.

macOS options:

- Finder: `Go -> Connect to Server...` and enter `http://127.0.0.1:8080/`
- Terminal with the built-in WebDAV client:

```bash
mkdir -p ~/mnt/clatterdrive
mount_webdav http://127.0.0.1:8080/ ~/mnt/clatterdrive
cp /path/to/local-file ~/mnt/clatterdrive/
cat ~/mnt/clatterdrive/local-file
```

macOS note:

- the WSL `davfs2` flow above was verified on this machine; the macOS commands use the standard built-in `mount_webdav` path but were not executed here because this host is Windows.

If live audio is enabled, directory creation and listing, uploads, downloads, overwrites, deletes, fragmented reads, and wake-from-standby activity all feed the audio model.

## Useful Environment Variables

- `FAKE_HDD_HOST`: bind to a different interface
- `FAKE_HDD_PORT`: use a different port
- `FAKE_HDD_BACKING_DIR`: use a different backing directory
- `FAKE_HDD_AUDIO=off`: disable live audio
- `FAKE_HDD_AUDIO_DEVICE`: pick an explicit output device index/name for PortAudio
- `FAKE_HDD_AUDIO_TEE_PATH`: record rendered output to a WAV, even when live audio is off
- `FAKE_HDD_EVENT_TRACE_PATH`: export a structured storage-event JSON trace on shutdown
- `FAKE_HDD_TRACE_EVENTS=on`: print compact event debug lines to stderr
- `FAKE_HDD_COLD_START=off`: start already ready
- `FAKE_HDD_ASYNC_POWER_ON=off`: disable background startup sequencing
- `FAKE_HDD_DRIVE_PROFILE`: choose a drive preset
- `FAKE_HDD_ACOUSTIC_PROFILE`: choose an installation/acoustic preset

Current drive presets:

- `desktop_7200_internal`
- `archive_5900_internal`
- `enterprise_7200_bare`
- `wd_ultrastar_hc550`
- `external_usb_enclosure`

Current acoustic presets:

- `bare_drive_lab`
- `mounted_in_case`
- `external_enclosure`
- `drive_on_desk`

Example:

```powershell
$env:FAKE_HDD_DRIVE_PROFILE = "archive_5900_internal"
$env:FAKE_HDD_ACOUSTIC_PROFILE = "drive_on_desk"
uv run python main.py
```

## Docker

Docker is mainly for the headless WebDAV/latency path. Native host execution is better for live audio.

```powershell
docker compose up --build
```

Or:

```powershell
docker build -t clatterdrive .
docker run --rm -p 8080:8080 -e FAKE_HDD_AUDIO=off -v "${PWD}/backing_storage:/data" clatterdrive
```

Audible container output needs an explicit host-audio bridge; Docker does not expose your speakers automatically. The supported path is a host PulseAudio/PipeWire server that the container can reach via `PULSE_SERVER`.

Example:

```powershell
$env:FAKE_HDD_AUDIO = "live"
$env:PULSE_SERVER = "host.docker.internal"
docker compose up --build
```

If your host audio server requires a specific PortAudio device, also set:

```powershell
$env:FAKE_HDD_AUDIO_DEVICE = "0"
```

If you do not already have a PulseAudio/PipeWire network endpoint on the host, use the native host run instead of Docker for audible playback.

Headless Docker smoke test for WebDAV plus rendered audio artifacts:

```powershell
uv run python -m tools.docker_webdav_audio_smoke
```

This writes temporary artifacts under `.runtime/docker-e2e/`, uploads and downloads through WebDAV, then verifies both the event trace and tee WAV are nonempty.
It also exercises large transfer, many-small-file, fragmented, and large-directory-listing workloads.

## Repo Layout

- [clatterdrive](clatterdrive): packaged application code
- [tools](tools): repo-local generators, profilers, trace exporters, audits, and fitting utilities
- [samples](samples): checked-in demo WAVs
- [docs](docs): GitHub Pages demo assets plus the internal MH tuning lab
- [tests](tests): test suite

Key files:

- [main.py](main.py): thin startup entrypoint
- [clatterdrive/app.py](clatterdrive/app.py): server startup and wiring
- [clatterdrive/webdav/provider.py](clatterdrive/webdav/provider.py): WebDAV interception layer
- [clatterdrive/hdd/latency.py](clatterdrive/hdd/latency.py): HDD timing and power-state model
- [clatterdrive/audio/core.py](clatterdrive/audio/core.py): audio plant and render logic
- [clatterdrive/audio/physics.py](clatterdrive/audio/physics.py): labeled physical-state, plausible-model, and artistic-calibration audio primitives
- [clatterdrive/audio/engine.py](clatterdrive/audio/engine.py): runtime audio shell

## Development

Smoke test:

```powershell
uv run python smoke.py
```

Tests:

```powershell
uv run python -m pytest -q
```

Regenerate the public sample clips:

```powershell
uv run python -m tools.generate_readme_demo_samples
```

Audio traces and audits:

```powershell
uv run python -m tools.trace_audio_scenarios
uv run python -m tools.audit_audio_stack
```

Internal calibration tooling:

```powershell
uv run python -m tools.fit_mh_reference
```

Notes:

- `tools.fit_mh_reference` is internal calibration tooling for the MH thrash lab, not part of the normal runtime path.
- It requires a local reference bundle under `.runtime/local_refs/` that is intentionally not tracked in git.

## Limits

- This is not a real block device or kernel filesystem.
- Out-of-band edits to the backing tree are only partially reconciled.
- WebDAV locks are in-memory and exist to satisfy clients like the Windows WebClient; they are not persisted across restart.
- The project is trying to sound cool and feel plausible, not to be a reference-grade HDD/acoustics model.
- The audio model is explicitly labeled by tier: physical state covers runtime variables such as spindle phase/RPM and actuator position/velocity; plausible model covers shapes such as motor lag, seek profiles, servo control, and modal resonators; artistic calibration covers coupling weights, gain curves, and output shaping chosen for sound.

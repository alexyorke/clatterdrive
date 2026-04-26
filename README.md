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

Example:

```powershell
curl.exe -X MKCOL http://127.0.0.1:8080/demo/
curl.exe -T "C:\path\to\file.bin" http://127.0.0.1:8080/demo/file.bin
curl.exe http://127.0.0.1:8080/demo/file.bin --output "C:\path\to\copy.bin"
```

If live audio is enabled, directory creation and listing, uploads, downloads, overwrites, deletes, fragmented reads, and wake-from-standby activity all feed the audio model.

## Useful Environment Variables

- `FAKE_HDD_HOST`: bind to a different interface
- `FAKE_HDD_PORT`: use a different port
- `FAKE_HDD_BACKING_DIR`: use a different backing directory
- `FAKE_HDD_AUDIO=off`: disable live audio
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

## Limits

- This is not a real block device or kernel filesystem.
- Out-of-band edits to the backing tree are only partially reconciled.
- WebDAV locking is intentionally rejected instead of being simulated as media traffic.
- The project is trying to sound cool and feel plausible, not to be a reference-grade HDD/acoustics model.

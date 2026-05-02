from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .app import start_server
from .config import ClatterDriveConfig, config_from_env, doctor_report, profile_catalog, report_as_json


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--backing-dir", default=None)
    parser.add_argument("--audio", choices=("live", "off"), default=None)
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--audio-tee-path", default=None)
    parser.add_argument("--event-trace-path", default=None)
    parser.add_argument("--trace-events", action="store_true")
    parser.add_argument("--drive-profile", default=None)
    parser.add_argument("--acoustic-profile", default=None)
    parser.add_argument("--ready", action="store_true", help="Start with the simulated drive already ready.")
    parser.add_argument("--sync-power-on", action="store_true", help="Disable background startup sequencing.")


def _config_from_args(args: argparse.Namespace) -> ClatterDriveConfig:
    base = config_from_env()
    return ClatterDriveConfig(
        host=args.host or base.host,
        port=base.port if args.port is None else int(args.port),
        backing_dir=args.backing_dir or base.backing_dir,
        audio=args.audio or base.audio,
        audio_device=args.audio_device or base.audio_device,
        audio_tee_path=args.audio_tee_path or base.audio_tee_path,
        event_trace_path=args.event_trace_path or base.event_trace_path,
        trace_events=bool(args.trace_events or base.trace_events),
        cold_start=False if args.ready else base.cold_start,
        async_power_on=False if args.sync_power_on else base.async_power_on,
        drive_profile=args.drive_profile or base.drive_profile,
        acoustic_profile=args.acoustic_profile or base.acoustic_profile,
    )


def _print_text_doctor(report: dict[str, object]) -> None:
    print(f"ClatterDrive doctor: {'ok' if report['ok'] else 'needs attention'}")
    checks = report.get("checks", {})
    if isinstance(checks, dict):
        for name, check in checks.items():
            if isinstance(check, dict):
                status = "ok" if check.get("ok") else "fail"
                print(f"- {name}: {status} - {check.get('message', '')}")
    warnings = report.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            print(f"warning: {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ClatterDrive local WebDAV HDD simulator.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the WebDAV/audio backend.")
    _add_config_args(serve)
    serve.add_argument("--json-status", action="store_true", help="Print one JSON readiness line after startup.")

    profiles = subparsers.add_parser("profiles", help="List drive and acoustic profiles.")
    profiles.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Validate local ClatterDrive configuration.")
    _add_config_args(doctor)
    doctor.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    command = args.command or "serve"
    if command == "serve":
        if args.command is None:
            args = parser.parse_args(["serve", *(sys.argv[1:] if argv is None else argv)])
        start_server(_config_from_args(args), json_status=bool(args.json_status))
        return
    if command == "profiles":
        catalog = profile_catalog()
        if args.json:
            print(report_as_json(catalog))
        else:
            print("Drive profiles:")
            for profile in catalog["drive_profiles"]:
                print(f"- {profile['name']}: {profile['description']}")
            print("Acoustic profiles:")
            for profile in catalog["acoustic_profiles"]:
                print(f"- {profile['name']}: {profile['description']}")
        return
    if command == "doctor":
        report = doctor_report(_config_from_args(args))
        if args.json:
            print(report_as_json(report))
        else:
            _print_text_doctor(report)
        if not report["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

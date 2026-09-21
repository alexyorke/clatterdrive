from __future__ import annotations

from dataclasses import dataclass


ModeDefinitions = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class DriveModalCalibration:
    calibration_id: str
    evidence: str
    reference_bucket: str | None
    reference_ids: tuple[str, ...]
    base_modes: ModeDefinitions
    cover_modes: ModeDefinitions
    actuator_modes: ModeDefinitions


_DESKTOP_MODES = DriveModalCalibration(
    calibration_id="desktop-engineering-v1",
    evidence="engineering_prior",
    reference_bucket="desktop_7200_internal",
    reference_ids=("open-cover-7200rpm", "awesome-seeking-sound"),
    base_modes=((72.0, 0.095, 0.92), (118.0, 0.082, 0.74), (168.0, 0.070, 0.56), (248.0, 0.060, 0.34)),
    cover_modes=(
        (212.0, 0.055, 0.44),
        (412.0, 0.048, 0.28),
        (576.0, 0.042, 0.24),
        (822.0, 0.038, 0.18),
        (1208.0, 0.034, 0.12),
    ),
    actuator_modes=((980.0, 0.036, 0.40), (1325.0, 0.032, 0.56), (1680.0, 0.028, 0.34)),
)

_ARCHIVE_MODES = DriveModalCalibration(
    calibration_id="archive-engineering-v1",
    evidence="engineering_prior",
    reference_bucket=None,
    reference_ids=(),
    base_modes=((66.0, 0.105, 0.94), (108.0, 0.090, 0.76), (154.0, 0.078, 0.54), (226.0, 0.067, 0.30)),
    cover_modes=((190.0, 0.064, 0.42), (360.0, 0.056, 0.27), (510.0, 0.050, 0.21), (748.0, 0.044, 0.15)),
    actuator_modes=((840.0, 0.044, 0.36), (1160.0, 0.038, 0.48), (1490.0, 0.034, 0.28)),
)

_ENTERPRISE_MODES = DriveModalCalibration(
    calibration_id="enterprise-ultrastar-reference-v1",
    evidence="reference_guided",
    reference_bucket="enterprise_ultrastar",
    reference_ids=("hc520-helium-startup", "hc550-spinup-down", "hc530-spinup", "hc530-helium-vs-normal"),
    base_modes=((78.0, 0.090, 0.86), (126.0, 0.078, 0.70), (184.0, 0.066, 0.58), (270.0, 0.056, 0.38)),
    cover_modes=(
        (228.0, 0.052, 0.42),
        (438.0, 0.045, 0.31),
        (612.0, 0.040, 0.27),
        (874.0, 0.036, 0.20),
        (1280.0, 0.032, 0.14),
    ),
    actuator_modes=((1040.0, 0.034, 0.42), (1410.0, 0.030, 0.58), (1810.0, 0.026, 0.36)),
)

_EXTERNAL_MODES = DriveModalCalibration(
    calibration_id="external-enclosure-engineering-v1",
    evidence="engineering_prior",
    reference_bucket="external_enclosure",
    reference_ids=(),
    base_modes=((62.0, 0.115, 0.88), (102.0, 0.102, 0.72), (148.0, 0.090, 0.50), (216.0, 0.078, 0.28)),
    cover_modes=((184.0, 0.074, 0.38), (344.0, 0.066, 0.25), (492.0, 0.058, 0.18), (710.0, 0.052, 0.12)),
    actuator_modes=((820.0, 0.050, 0.32), (1100.0, 0.044, 0.42), (1420.0, 0.038, 0.24)),
)


DRIVE_MODAL_CALIBRATIONS: dict[str, DriveModalCalibration] = {
    "desktop_7200_internal": _DESKTOP_MODES,
    "archive_5900_internal": _ARCHIVE_MODES,
    "enterprise_7200_bare": _ENTERPRISE_MODES,
    "wd_ultrastar_hc550": _ENTERPRISE_MODES,
    "seagate_ironwolf_pro_16tb": _ENTERPRISE_MODES,
    "external_usb_enclosure": _EXTERNAL_MODES,
}


def resolve_drive_modal_calibration(drive_profile_name: str) -> DriveModalCalibration:
    return DRIVE_MODAL_CALIBRATIONS.get(drive_profile_name, _DESKTOP_MODES)


__all__ = ["DRIVE_MODAL_CALIBRATIONS", "DriveModalCalibration", "resolve_drive_modal_calibration"]

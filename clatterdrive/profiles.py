from __future__ import annotations

from dataclasses import dataclass

from .runtime.deps import EnvReader, OSEnvReader


@dataclass(frozen=True)
class DriveProfile:
    """Drive timing plus audio calibration.

    Physical state inputs: `rpm`, `platters`, block/cache sizes, and power/seek
    timings describe the simulated drive. Plausible model inputs: seek, spin,
    transfer, and command timing terms drive latency and motion. Artistic
    calibration inputs: harmonic weights, frequency scales, and audio gains tune
    rendered sound and are not measured hardware constants. Profiles may also
    reference a source-backed hardware prior for derived physical constants.
    """

    name: str
    description: str
    default_acoustic_profile: str
    rpm: int
    platters: int
    avg_seek_ms: float
    track_to_track_ms: float
    settle_ms: float
    head_switch_ms: float
    transfer_rate_outer_mbps: float
    transfer_rate_inner_mbps: float
    ncq_depth: int
    read_ahead_kb: int
    write_cache_mb: int
    dirty_expire_ms: float
    standby_after_s: float
    unload_after_s: float
    low_rpm_after_s: float
    spinup_ms: float
    standby_to_ready_ms: float
    power_on_to_ready_ms: float
    unload_to_ready_ms: float
    low_rpm_to_ready_ms: float
    low_rpm_rpm: int
    spin_down_ms: float
    ready_poll_ms: float
    identify_poll_ms: float
    test_unit_ready_ms: float
    command_overhead_ms: float
    command_overheads_by_kind: tuple[tuple[str, float], ...]
    queue_depth_penalty_ms: float
    spindle_harmonics: tuple[int, ...]
    spindle_harmonic_weights: tuple[float, ...]
    platter_frequency_scale: float
    cover_frequency_scale: float
    actuator_frequency_scale: float
    platter_gain_scale: float
    cover_gain_scale: float
    actuator_gain_scale: float
    windage_gain: float
    bearing_gain: float
    boundary_excitation_gain: float
    helium: bool = False
    hardware_prior: str | None = None
    spindle_inertia_scale: float = 1.0
    windage_drag_share_at_nominal: float = 0.04


@dataclass(frozen=True)
class AcousticProfile:
    """Mount/acoustic sound-design profile.

    These fields are artistic calibration around a plausible structure-borne and
    airborne acoustic model: coupling, resonance, damping, radiation, filtering,
    and output gains describe how the installation should sound, not a measured
    CAD/acoustics transfer function.
    """

    name: str
    description: str
    output_gain: float
    direct_gain: float
    platter_gain: float
    cover_gain: float
    actuator_gain: float
    structure_gain: float
    desk_coupling: float
    enclosure_coupling: float
    internal_air_coupling: float
    mount_damping_scale: float
    enclosure_mass_scale: float
    enclosure_resonance_scale: float
    enclosure_radiation_gain: float
    table_mass_scale: float
    table_resonance_scale: float
    table_damping_scale: float
    table_radiation_gain: float
    impulse_gain: float
    sequential_boundary_gain: float
    final_lowpass_hz: float
    final_highpass_hz: float


DRIVE_PROFILES: dict[str, DriveProfile] = {
    "desktop_7200_internal": DriveProfile(
        name="desktop_7200_internal",
        description="General desktop 7200 RPM internal drive mounted in a normal PC.",
        default_acoustic_profile="mounted_in_case",
        rpm=7200,
        platters=3,
        avg_seek_ms=8.4,
        track_to_track_ms=0.25,
        settle_ms=0.35,
        head_switch_ms=0.35,
        transfer_rate_outer_mbps=210.0,
        transfer_rate_inner_mbps=120.0,
        ncq_depth=32,
        read_ahead_kb=512,
        write_cache_mb=32,
        dirty_expire_ms=350.0,
        standby_after_s=60.0,
        unload_after_s=12.0,
        low_rpm_after_s=30.0,
        spinup_ms=3200.0,
        standby_to_ready_ms=9000.0,
        power_on_to_ready_ms=11250.0,
        unload_to_ready_ms=900.0,
        low_rpm_to_ready_ms=4200.0,
        low_rpm_rpm=6300,
        spin_down_ms=2400.0,
        ready_poll_ms=24.0,
        identify_poll_ms=0.35,
        test_unit_ready_ms=0.18,
        command_overhead_ms=0.08,
        command_overheads_by_kind=(
            ("metadata", 0.08),
            ("journal", 0.14),
            ("data", 0.08),
            ("writeback", 0.11),
            ("flush", 0.22),
            ("background", 0.10),
        ),
        queue_depth_penalty_ms=0.035,
        spindle_harmonics=(1, 2, 3, 4),
        spindle_harmonic_weights=(1.0, 0.42, 0.23, 0.14),
        platter_frequency_scale=1.0,
        cover_frequency_scale=1.0,
        actuator_frequency_scale=1.0,
        platter_gain_scale=1.0,
        cover_gain_scale=1.0,
        actuator_gain_scale=1.0,
        windage_gain=1.0,
        bearing_gain=1.0,
        boundary_excitation_gain=1.0,
    ),
    "archive_5900_internal": DriveProfile(
        name="archive_5900_internal",
        description="Slower 5900 RPM archive-style internal drive with gentler access behavior.",
        default_acoustic_profile="mounted_in_case",
        rpm=5900,
        platters=4,
        avg_seek_ms=11.4,
        track_to_track_ms=0.38,
        settle_ms=0.42,
        head_switch_ms=0.40,
        transfer_rate_outer_mbps=185.0,
        transfer_rate_inner_mbps=105.0,
        ncq_depth=32,
        read_ahead_kb=768,
        write_cache_mb=64,
        dirty_expire_ms=425.0,
        standby_after_s=75.0,
        unload_after_s=16.0,
        low_rpm_after_s=42.0,
        spinup_ms=3800.0,
        standby_to_ready_ms=10400.0,
        power_on_to_ready_ms=12600.0,
        unload_to_ready_ms=1080.0,
        low_rpm_to_ready_ms=5000.0,
        low_rpm_rpm=5000,
        spin_down_ms=2800.0,
        ready_poll_ms=28.0,
        identify_poll_ms=0.42,
        test_unit_ready_ms=0.22,
        command_overhead_ms=0.10,
        command_overheads_by_kind=(
            ("metadata", 0.09),
            ("journal", 0.15),
            ("data", 0.10),
            ("writeback", 0.12),
            ("flush", 0.24),
            ("background", 0.11),
        ),
        queue_depth_penalty_ms=0.05,
        spindle_harmonics=(1, 2, 3, 4),
        spindle_harmonic_weights=(1.0, 0.36, 0.18, 0.10),
        platter_frequency_scale=0.92,
        cover_frequency_scale=0.94,
        actuator_frequency_scale=0.90,
        platter_gain_scale=0.92,
        cover_gain_scale=0.94,
        actuator_gain_scale=0.88,
        windage_gain=0.82,
        bearing_gain=0.95,
        boundary_excitation_gain=0.78,
    ),
    "enterprise_7200_bare": DriveProfile(
        name="enterprise_7200_bare",
        description="Faster 7200 RPM enterprise drive with stronger direct radiation and busier queue behavior.",
        default_acoustic_profile="bare_drive_lab",
        rpm=7200,
        platters=5,
        avg_seek_ms=7.2,
        track_to_track_ms=0.19,
        settle_ms=0.30,
        head_switch_ms=0.28,
        transfer_rate_outer_mbps=255.0,
        transfer_rate_inner_mbps=150.0,
        ncq_depth=64,
        read_ahead_kb=1024,
        write_cache_mb=128,
        dirty_expire_ms=250.0,
        standby_after_s=180.0,
        unload_after_s=20.0,
        low_rpm_after_s=60.0,
        spinup_ms=3400.0,
        standby_to_ready_ms=8200.0,
        power_on_to_ready_ms=10600.0,
        unload_to_ready_ms=760.0,
        low_rpm_to_ready_ms=3900.0,
        low_rpm_rpm=6450,
        spin_down_ms=2300.0,
        ready_poll_ms=20.0,
        identify_poll_ms=0.30,
        test_unit_ready_ms=0.15,
        command_overhead_ms=0.07,
        command_overheads_by_kind=(
            ("metadata", 0.07),
            ("journal", 0.12),
            ("data", 0.07),
            ("writeback", 0.09),
            ("flush", 0.18),
            ("background", 0.08),
        ),
        queue_depth_penalty_ms=0.07,
        spindle_harmonics=(1, 2, 3, 4, 5),
        spindle_harmonic_weights=(1.0, 0.48, 0.27, 0.17, 0.08),
        platter_frequency_scale=1.04,
        cover_frequency_scale=1.03,
        actuator_frequency_scale=1.08,
        platter_gain_scale=1.18,
        cover_gain_scale=1.12,
        actuator_gain_scale=1.16,
        windage_gain=1.20,
        bearing_gain=1.08,
        boundary_excitation_gain=1.28,
    ),
    "wd_ultrastar_hc550": DriveProfile(
        name="wd_ultrastar_hc550",
        description="WD Ultrastar DC HC550-class 7200 RPM helium enterprise drive.",
        default_acoustic_profile="bare_drive_lab",
        rpm=7200,
        platters=9,
        avg_seek_ms=7.7,
        track_to_track_ms=0.18,
        settle_ms=0.28,
        head_switch_ms=0.26,
        transfer_rate_outer_mbps=255.0,
        transfer_rate_inner_mbps=150.0,
        ncq_depth=64,
        read_ahead_kb=1024,
        write_cache_mb=512,
        dirty_expire_ms=220.0,
        standby_after_s=180.0,
        unload_after_s=22.0,
        low_rpm_after_s=70.0,
        spinup_ms=11800.0,
        standby_to_ready_ms=15000.0,
        power_on_to_ready_ms=20000.0,
        unload_to_ready_ms=1000.0,
        low_rpm_to_ready_ms=4000.0,
        low_rpm_rpm=6300,
        spin_down_ms=3200.0,
        ready_poll_ms=24.0,
        identify_poll_ms=0.30,
        test_unit_ready_ms=0.15,
        command_overhead_ms=0.07,
        command_overheads_by_kind=(
            ("metadata", 0.07),
            ("journal", 0.12),
            ("data", 0.07),
            ("writeback", 0.09),
            ("flush", 0.18),
            ("background", 0.08),
        ),
        queue_depth_penalty_ms=0.075,
        spindle_harmonics=(1, 2, 3, 4, 5),
        spindle_harmonic_weights=(1.0, 0.50, 0.30, 0.18, 0.09),
        platter_frequency_scale=1.06,
        cover_frequency_scale=1.04,
        actuator_frequency_scale=1.10,
        platter_gain_scale=1.24,
        cover_gain_scale=1.14,
        actuator_gain_scale=1.18,
        windage_gain=1.28,
        bearing_gain=1.12,
        boundary_excitation_gain=1.34,
        helium=True,
    ),
    "seagate_ironwolf_pro_16tb": DriveProfile(
        name="seagate_ironwolf_pro_16tb",
        description="Seagate IronWolf Pro 16TB ST16000NT001/ST16000NTZ01 7200 RPM NAS drive.",
        default_acoustic_profile="mounted_in_case",
        rpm=7200,
        platters=8,
        avg_seek_ms=8.2,
        track_to_track_ms=0.22,
        settle_ms=0.32,
        head_switch_ms=0.30,
        transfer_rate_outer_mbps=285.0,
        transfer_rate_inner_mbps=150.0,
        ncq_depth=32,
        read_ahead_kb=1024,
        write_cache_mb=512,
        dirty_expire_ms=260.0,
        standby_after_s=900.0,
        unload_after_s=120.0,
        low_rpm_after_s=600.0,
        spinup_ms=25000.0,
        standby_to_ready_ms=25000.0,
        power_on_to_ready_ms=25000.0,
        unload_to_ready_ms=950.0,
        low_rpm_to_ready_ms=4800.0,
        low_rpm_rpm=6300,
        spin_down_ms=20000.0,
        ready_poll_ms=30.0,
        identify_poll_ms=0.30,
        test_unit_ready_ms=0.16,
        command_overhead_ms=0.08,
        command_overheads_by_kind=(
            ("metadata", 0.08),
            ("journal", 0.13),
            ("data", 0.08),
            ("writeback", 0.10),
            ("flush", 0.20),
            ("background", 0.10),
        ),
        queue_depth_penalty_ms=0.055,
        spindle_harmonics=(1, 2, 3, 4, 5),
        spindle_harmonic_weights=(1.0, 0.46, 0.25, 0.15, 0.07),
        platter_frequency_scale=1.02,
        cover_frequency_scale=1.01,
        actuator_frequency_scale=1.05,
        platter_gain_scale=1.08,
        cover_gain_scale=1.05,
        actuator_gain_scale=1.06,
        windage_gain=0.96,
        bearing_gain=1.02,
        boundary_excitation_gain=1.10,
        helium=True,
        hardware_prior="seagate_ironwolf_pro_16tb",
        spindle_inertia_scale=1.1666945025797724,
        windage_drag_share_at_nominal=0.3397498862622795,
    ),
    "external_usb_enclosure": DriveProfile(
        name="external_usb_enclosure",
        description="Consumer external USB drive with bridge overhead and softer internal mechanics.",
        default_acoustic_profile="external_enclosure",
        rpm=5400,
        platters=2,
        avg_seek_ms=13.0,
        track_to_track_ms=0.45,
        settle_ms=0.44,
        head_switch_ms=0.42,
        transfer_rate_outer_mbps=165.0,
        transfer_rate_inner_mbps=95.0,
        ncq_depth=16,
        read_ahead_kb=512,
        write_cache_mb=32,
        dirty_expire_ms=460.0,
        standby_after_s=50.0,
        unload_after_s=10.0,
        low_rpm_after_s=24.0,
        spinup_ms=3300.0,
        standby_to_ready_ms=9600.0,
        power_on_to_ready_ms=11800.0,
        unload_to_ready_ms=950.0,
        low_rpm_to_ready_ms=4300.0,
        low_rpm_rpm=4700,
        spin_down_ms=2500.0,
        ready_poll_ms=30.0,
        identify_poll_ms=0.48,
        test_unit_ready_ms=0.25,
        command_overhead_ms=0.14,
        command_overheads_by_kind=(
            ("metadata", 0.13),
            ("journal", 0.18),
            ("data", 0.14),
            ("writeback", 0.17),
            ("flush", 0.28),
            ("background", 0.16),
        ),
        queue_depth_penalty_ms=0.055,
        spindle_harmonics=(1, 2, 3, 4),
        spindle_harmonic_weights=(1.0, 0.34, 0.16, 0.08),
        platter_frequency_scale=0.88,
        cover_frequency_scale=0.90,
        actuator_frequency_scale=0.92,
        platter_gain_scale=0.88,
        cover_gain_scale=1.02,
        actuator_gain_scale=0.85,
        windage_gain=0.76,
        bearing_gain=0.92,
        boundary_excitation_gain=0.72,
    ),
}


ACOUSTIC_PROFILES: dict[str, AcousticProfile] = {
    "bare_drive_lab": AcousticProfile(
        name="bare_drive_lab",
        description="Bare drive on soft spacers in a measurement-like setup.",
        output_gain=0.98,
        direct_gain=0.72,
        platter_gain=1.04,
        cover_gain=1.00,
        actuator_gain=1.14,
        structure_gain=0.46,
        desk_coupling=0.06,
        enclosure_coupling=0.08,
        internal_air_coupling=0.04,
        mount_damping_scale=0.70,
        enclosure_mass_scale=0.25,
        enclosure_resonance_scale=1.30,
        enclosure_radiation_gain=0.10,
        table_mass_scale=0.55,
        table_resonance_scale=1.10,
        table_damping_scale=1.15,
        table_radiation_gain=0.10,
        impulse_gain=1.12,
        sequential_boundary_gain=1.00,
        final_lowpass_hz=6200.0,
        final_highpass_hz=35.0,
    ),
    "mounted_in_case": AcousticProfile(
        name="mounted_in_case",
        description="Internal drive heard through a normal PC case and mounting path.",
        output_gain=0.78,
        direct_gain=0.46,
        platter_gain=0.88,
        cover_gain=1.06,
        actuator_gain=0.88,
        structure_gain=0.72,
        desk_coupling=0.32,
        enclosure_coupling=1.15,
        internal_air_coupling=0.62,
        mount_damping_scale=1.35,
        enclosure_mass_scale=2.20,
        enclosure_resonance_scale=0.82,
        enclosure_radiation_gain=1.15,
        table_mass_scale=2.40,
        table_resonance_scale=0.74,
        table_damping_scale=0.92,
        table_radiation_gain=1.45,
        impulse_gain=0.82,
        sequential_boundary_gain=0.90,
        final_lowpass_hz=3600.0,
        final_highpass_hz=55.0,
    ),
    "external_enclosure": AcousticProfile(
        name="external_enclosure",
        description="Consumer external enclosure with extra shell damping and bridge/noise isolation.",
        output_gain=0.72,
        direct_gain=0.38,
        platter_gain=0.80,
        cover_gain=1.08,
        actuator_gain=0.82,
        structure_gain=0.80,
        desk_coupling=0.18,
        enclosure_coupling=0.92,
        internal_air_coupling=0.78,
        mount_damping_scale=1.55,
        enclosure_mass_scale=1.55,
        enclosure_resonance_scale=0.76,
        enclosure_radiation_gain=0.82,
        table_mass_scale=1.25,
        table_resonance_scale=0.84,
        table_damping_scale=1.20,
        table_radiation_gain=0.95,
        impulse_gain=0.74,
        sequential_boundary_gain=0.80,
        final_lowpass_hz=3000.0,
        final_highpass_hz=60.0,
    ),
    "drive_on_desk": AcousticProfile(
        name="drive_on_desk",
        description="Drive or enclosure coupled directly into a desk surface.",
        output_gain=0.64,
        direct_gain=0.48,
        platter_gain=0.90,
        cover_gain=1.02,
        actuator_gain=0.90,
        structure_gain=0.82,
        desk_coupling=0.44,
        enclosure_coupling=0.42,
        internal_air_coupling=0.24,
        mount_damping_scale=1.14,
        enclosure_mass_scale=1.10,
        enclosure_resonance_scale=0.92,
        enclosure_radiation_gain=0.34,
        table_mass_scale=3.20,
        table_resonance_scale=0.78,
        table_damping_scale=1.18,
        table_radiation_gain=0.88,
        impulse_gain=0.82,
        sequential_boundary_gain=0.96,
        final_lowpass_hz=3900.0,
        final_highpass_hz=48.0,
    ),
}


def _profile_key(name: str) -> str:
    return name.strip().lower()


def resolve_drive_profile(profile: str | DriveProfile | None) -> DriveProfile:
    if isinstance(profile, DriveProfile):
        return profile

    profile_name = profile or "desktop_7200_internal"
    key = _profile_key(profile_name)
    if key not in DRIVE_PROFILES:
        available = ", ".join(sorted(DRIVE_PROFILES))
        raise ValueError(f"unknown drive profile {profile_name!r}; expected one of: {available}")
    return DRIVE_PROFILES[key]


def resolve_acoustic_profile(profile: str | AcousticProfile | None, *, drive_profile: DriveProfile | None = None) -> AcousticProfile:
    if isinstance(profile, AcousticProfile):
        return profile

    resolved_drive = resolve_drive_profile(drive_profile) if not isinstance(drive_profile, DriveProfile) else drive_profile
    default_name = resolved_drive.default_acoustic_profile if resolved_drive is not None else "mounted_in_case"
    profile_name = profile or default_name
    key = _profile_key(profile_name)
    if key not in ACOUSTIC_PROFILES:
        available = ", ".join(sorted(ACOUSTIC_PROFILES))
        raise ValueError(f"unknown acoustic profile {profile_name!r}; expected one of: {available}")
    return ACOUSTIC_PROFILES[key]


def resolve_selected_profiles(
    drive_profile: str | DriveProfile | None,
    acoustic_profile: str | AcousticProfile | None,
) -> tuple[DriveProfile, AcousticProfile]:
    resolved_drive = resolve_drive_profile(drive_profile)
    resolved_acoustic = resolve_acoustic_profile(acoustic_profile, drive_profile=resolved_drive)
    return resolved_drive, resolved_acoustic


def resolve_drive_profile_from_env(
    profile: str | DriveProfile | None,
    *,
    env: EnvReader | None = None,
) -> DriveProfile:
    env_reader = env or OSEnvReader()
    profile_name = profile
    if not isinstance(profile_name, DriveProfile):
        profile_name = profile_name or env_reader.get("FAKE_HDD_DRIVE_PROFILE", "desktop_7200_internal")
    return resolve_drive_profile(profile_name)


def resolve_acoustic_profile_from_env(
    profile: str | AcousticProfile | None,
    *,
    drive_profile: str | DriveProfile | None = None,
    env: EnvReader | None = None,
) -> AcousticProfile:
    env_reader = env or OSEnvReader()
    resolved_drive = (
        drive_profile if isinstance(drive_profile, DriveProfile) else resolve_drive_profile_from_env(drive_profile, env=env_reader)
    )
    profile_name = profile
    if not isinstance(profile_name, AcousticProfile):
        profile_name = profile_name or env_reader.get(
            "FAKE_HDD_ACOUSTIC_PROFILE",
            resolved_drive.default_acoustic_profile,
        )
    return resolve_acoustic_profile(profile_name, drive_profile=resolved_drive)


def resolve_selected_profiles_from_env(
    drive_profile: str | DriveProfile | None,
    acoustic_profile: str | AcousticProfile | None,
    *,
    env: EnvReader | None = None,
) -> tuple[DriveProfile, AcousticProfile]:
    env_reader = env or OSEnvReader()
    resolved_drive = resolve_drive_profile_from_env(drive_profile, env=env_reader)
    resolved_acoustic = resolve_acoustic_profile_from_env(
        acoustic_profile,
        drive_profile=resolved_drive,
        env=env_reader,
    )
    return resolved_drive, resolved_acoustic

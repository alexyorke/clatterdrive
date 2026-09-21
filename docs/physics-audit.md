# Runtime physics audit, 2026-09-19

This is a source-backed consistency audit, not a claim of measured acoustic
accuracy. All 31 runtime Python modules (approximately 10,900 lines before
edits) were read in full. Launcher profile/settings boundaries, relevant tests,
and audio tools were inspected selectively; their entire source trees were not
audited line by line.

## Coverage ledger

| Partition | Modules | Evidence and disposition |
| --- | ---: | --- |
| Audio | 7 | Full read; corrected rate dependence, stationary rotor force, parking completion, and physical event interpretation |
| HDD | 4 | Full read; corrected throughput units, transition ordering, cache telemetry, and read-ahead state |
| Profiles and hardware priors | 2 | Full read; corrected SATA queue depth and contact terminology |
| Scheduler and storage events | 3 | Full read; extended physical telemetry; scheduling approximations retained |
| Filesystem | 5 | Full read; prior persistence fixes retained; synthetic metadata model documented below |
| Frontends, app, config, entry points | 7 | Full read; prior block completion fixes retained; WebDAV workflow tested |
| Runtime | 3 | Full read; no additional physics correction identified |

Stopping condition: complete the runtime source inspection, fix reproducible
inconsistencies supported by physical constraints, and validate regressions.
Measurement-dependent unknowns remain explicit rather than being replaced by
invented precision. See also the [earlier reliability review](realism-reliability-notes.md).

## Physical evidence and corrections

- Seagate distinguishes unloaded heads at full spindle speed from standby with
  a stopped spindle. Background head unloading now preserves RPM. Parking now
  completes its motion and settling stages, reaches the parked state, and
  excites the mechanical stop. Cancelled power transitions cannot complete over
  a newer transition; telemetry is emitted after its elapsed time.
  [IronWolf product manual, section 2.5.4](https://www.seagate.com/content/dam/seagate/migrated-assets/www-content/product-content/ironwolf/en-us/docs/100804010b.pdf)
- Published sustained rates are MB/s, not MiB/s. Transfer times and inferred
  sectors per revolution now use decimal megabytes. This corrects the former
  4.86 percent rate overstatement without changing persisted capacity units.
  [IronWolf Pro product manual, table 4](https://www.seagate.com/content/dam/seagate/assets/products/nas-drives/ironwolf-pro-hard-drive/files/Seagate_IronWolf_Pro_SATA_Product_Manual_24-20-16-12TB_206815300B.pdf)
- SATA NCQ tags span 0 through 31. Two SATA profiles incorrectly allowed 64
  queued commands; both now use 32. Seek and rotational positioning remain
  separate latency terms.
  [Intel/Seagate NCQ white paper](https://www.seagate.com/docs/pdf/whitepaper/D2c_tech_paper_intc-stx_sata_ncq.pdf)
- Discrete resonator poles and filter decay depend on sample rate. Modal force
  increments now scale with the time step; noise filters and transient decays
  retain their reference time constants at other sample rates. Stopped rotors
  no longer produce periodic excitation. The legacy startup filter calibration
  remains referenced to 22,050 Hz; running filters use 44,100 Hz.
  [Smith, modal expansion](https://www.dsprelated.com/freebooks/pasp/Modal_Expansion.html),
  [Sachs, single-pole filter](https://www.dsprelated.com/showarticle/779/ten-little-algorithms-part-2-the-single-pole-low-pass-filter)
- Foreground physical events now carry signed radial displacement, absolute
  target position, positioning delay, and initial track fraction. Small seeks
  are not enlarged to an arbitrary minimum stroke. Actual transfer duration is
  retained, starts after positioning, and ends without a synthetic activity
  tail. A read satisfied entirely by cache emits no mechanical transfer event.
- The workload mapper no longer invents extra seeks and transfers for requests
  whose extents were already simulated. Physical commands no longer receive
  directory-size, fragmentation, queue-depth, or repetition-based force bias.
  Legacy hand-authored demo events retain their compatibility behavior.
- Adaptive read-ahead window state now survives the latency/cache wrapper.
  Repeated startup telemetry no longer restarts the startup envelope, and an
  access that already waited for readiness no longer requests another spinup.

## Validation and reference policy

New regressions exercise sample-rate consistency, stationary excitation,
parking, startup continuity, cancelled transitions, cache silence, signed
seeks, adaptive cache state, small seeks, long transfers, positioning delay,
decimal throughput, and SATA queue depth.

The two generated startup/power-cycle demo WAVs and their documentation copies
were refreshed. No real recording or metadata-storm golden was changed. Demo
spectral comparisons now encode both sides as PCM16: floating-point versus
16-bit quantization had changed magnitude-weighted spectral features despite
waveform agreement within one quantization step. Tolerances were not loosened.
Independent startup bounds and the unchanged metadata golden remain checks.

Final validation: 210 tests passed through `scripts/test.ps1`; the separate
audio benchmark passed all six startup acceptance checks, with metadata-golden
correlation 0.999605 and RMS difference 0.007083. `scripts/lint.ps1` passed
compile checks, Ruff, Vulture, and Mypy (66 source files). The source-backend
Windows WebDAV end-to-end check passed and produced nonempty audio and event
traces. Packaged launchers, mapped-drive integration, and speaker listening were
not validated in this pass.

## Remaining approximations and measurement gaps

- Servo force, mode gains, mounting response, and output level are normalized,
  not calibrated SI force or sound pressure. Some profile mass/gain descriptors
  are not used in the equations. A matching recording and measured transfer
  functions are needed for per-drive acoustic validation.
- Startup audio and storage readiness use distinct models. Readiness reconciles
  audio state, but accelerated storage latency is not a complete time-scaling
  of the acoustic model.
- Controller update cadence is not a vendor servo-wedge implementation. The
  sequential boundary proxy does not distinguish head switches from cylinder
  steps. Background and hand-authored events lack the full foreground position
  telemetry.
- Zone layout, skew, read-ahead speculation, cache lifetimes, background scans,
  and retries approximate firmware. Read-ahead does not account for every
  speculative physical read. Cache capacity is represented by limited spans.
- Filesystem directory growth, virtual blocks, writeback clustering, and journal
  behavior are workload approximations, not an on-disk filesystem replica or
  full crash-consistency model.
- Thermal behavior, detailed air-bearing dynamics, calibrated aerodynamic
  noise, and individual drive/mount tolerances require measurements beyond the
  available public specifications.

Sources were consulted on 2026-09-19. Only public links are retained; no remote
recording or reference bundle was downloaded during this audit.

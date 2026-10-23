# Code audit, 2026-10-22

Baseline: `a31a051` on `origin/master`. Work branch:
`codex/full-audit-2026-10-22`.

Scope: tracked, hand-maintained runtime, launcher, test, tool, build, and CI
source/configuration. Read each source file in full, trace suspected defects,
add regression tests, and run targeted checks followed by the repository gates.
Generated assets, recordings, dependency lockfile internals, and third-party
packages are excluded from line-by-line source review. Acoustic fidelity still
requires measurements; plausible equations alone do not establish calibration.

## Coverage

- Fully read: every module under `clatterdrive/`, `main.py`,
  `pyproject.toml`, `scripts/test.ps1`, `scripts/Use-RepoUv.ps1`,
  `tests/test_scheduler.py`, `tests/test_realism_audit.py`,
  `tests/test_physical_reliability.py`, `tests/conftest.py`.
- Also fully read: both launcher implementation trees, macOS launcher tests,
  Windows launcher view-model/mount tests, `tests/helpers.py`, `tests/test_app.py`,
  `tests/test_cli_config.py`, new audit regressions, `tools/backend_e2e.py`,
  `scripts/lint.ps1`, `scripts/test-ui.ps1`, `scripts/test-e2e.ps1`,
  `.github/workflows/ci-python.yml`, `.github/workflows/ci-macos.yml`.
- Continuation completed the remaining reads: every tracked file under `tests/`,
  `tools/`, `scripts/`, both launcher trees (including Windows UI smoke tests),
  and `.github/`; `smoke.py`, Dockerfile/Compose, ignore files, pytest settings,
  dotnet tool settings, reference manifest, and both hand-maintained HTML pages.
- Unit of coverage is the complete hand-maintained source/configuration file,
  inventoried with `git ls-files`. The source-reading stopping condition is met
  for this scope. Generated workload HTML/JSON, reference-summary JSON,
  recordings/images, and lockfile internals remain excluded. Reading coverage
  does not establish that every possible bug is fixed or every workflow tested.

## Findings to verify

- Shutdown during startup can leave readiness pollers waiting indefinitely.
- Idle power transitions can start while foreground disk I/O is active.
- Adjacent partial-block scheduler requests can merge into too few blocks.
- Scheduler position lookup exceptions can terminate the dispatch worker.
- Cache-hit reads bypass address validation, including speculative spans past
  the end of the disk.

## Validation

Baseline: 221 tests passed. The first sandboxed invocation failed on uv cache
permissions before running tests; it is not a product-test failure.

First ten added regressions failed against the baseline and now pass. Targeted
filesystem/scheduler/HDD gate: 67 passed. Power-state regression gate: 57 passed.
Writeback/virtual-drive gate: 46 passed. Final full Python gate: 245 passed in
59.08 seconds. Audio benchmark passed all six startup acceptance checks; metadata
correlation and RMS delta match the baseline. No audio goldens were changed.

## Corrections implemented so far

- Reject invalid physical spans before readiness/cache handling and cap read-ahead
  at media capacity. Preserve partial-block request coverage during scheduling.
- Reject unusable queue depths and survive advisory head-position lookup errors.
- Reject negative file ranges and make empty writes true no-ops.
- Reconcile removed descendants when scanning the volume root.
- Report startup failures and stopped-model accesses instead of hanging pollers.
- Serialize idle decisions with physical I/O; preserve host-idle timer origins
  across slowdown; give an expired standby timer precedence over calibration.
- Account for dequeued writebacks until completion, serialize drainers, retain
  failed batches, and surface background errors to sync before retry.
- Keep recording resources open if the headless rendering worker fails to stop;
  reject negative audio frame counts before advancing the render clock.
- Run remaining server cleanup even when an earlier cleanup raises.
- Count a merged physical dispatch once in latency/byte statistics.
- Preserve WsgiDAV root containment checks in custom destination resolution and
  preflight recursive COPY trees for symlinks/junctions before changing contents.
  This is not protection against concurrent changes by a local filesystem writer.
- Reject nonfinite launcher capacities; ignore non-status JSON in Windows backend
  output; use IPv6 loopback for shutdown of an IPv6 wildcard listener on Windows.

Physical/persistence references opened on 2026-10-22:

- [Seagate IronWolf Pro manual, Rev. D](https://www.seagate.com/files/www-content/product-content/ironwolf/en-us/docs/100835984d.pdf):
  unloaded-head and reduced-speed idle states are distinct from standby.
- [Linux VFS writeback error handling](https://docs.kernel.org/filesystems/vfs.html#handling-errors-during-writeback):
  synchronization must observe writeback completion and failures. The simulator
  retains failed batches for retry; it does not claim full kernel fsync semantics.

Additional review candidates, not yet resolved: stale optional environment
settings, startup head-load telemetry suppression, background-scan timing,
WebDAV property-tree URL normalization, and Windows launcher shutdown/UI-thread
coordination. Developer-tool follow-ups include deterministic offline trace
clocks, WAV format validation, and broader custom calibration-input validation.
These are follow-up candidates, not claimed fixes in this patch.

Additional verification: audio/app/regression suite 66 passed; Windows launcher
unit tests 11 passed; source backend E2E with space-containing paths passed
(MKCOL/PUT/GET/DELETE, WAV, event trace, graceful shutdown). No Swift toolchain
is installed on this Windows host, so the new Swift validation test is unrun.

One HTTP MOVE test timed out in two parallel runs, then passed in isolation,
12 consecutive standalone workflows, the 23-test parallel WebDAV suite, and the
final full suite. Thread dumping on HTTP test timeouts was added for diagnosis;
the timeout was not increased. A passing rerun alone does not establish the
cause is fixed.

## Continuation corrections and evidence

- Keep subprocess output draining after readiness, retain only a bounded log
  tail, and enforce readiness deadlines even when the child produces no output.
  Regression cases cover a silent process, non-object JSON, and a megabyte of
  post-readiness logging. This fixes a pipe-backpressure mechanism in the E2E
  harness; it is not evidence of the cause of the separate in-process MOVE flake.
- Reject normalized overlapping directory COPY/MOVE destinations before custom
  recursive operations can recurse into their own output or overwrite a source.
  HTTP tests verify rejected descendant destinations preserve source contents.
- Fail Windows build, restore, signing, installer, and test scripts on nonzero
  native exit codes, including Windows PowerShell 5.1. Mock-native failure-path
  tests cover bootstrap and the launcher build gate without installing software.
- Correct cross-correlation alignment direction and keep event weights attached
  to the windows that survived filtering. Zero-pad short analysis windows to
  match FFT/mel-bank geometry without changing the original duration. Six new
  tests reproduced wrong alignment and short-clip crashes before these fixes.
- Buffer macOS backend output as bytes through newline boundaries so split JSON
  and UTF-8 do not lose readiness. Drain profile output before waiting for exit;
  use IPv6 loopback when shutting down an IPv6 wildcard listener. Swift parser
  tests are included for native CI; Swift is unavailable on this Windows host.

Local continuation validation: 282 Python tests passed (61.64 seconds), including
24 repeated COPY/MOVE workflows; lint/type checks passed for 72 Python files;
11 Windows launcher unit tests passed; source E2E with space-containing paths
passed. No acceptance thresholds or audio goldens were relaxed. Native macOS,
Linux/Docker, packaged-app, and installer checks are delegated to the existing
CI for the pushed commit, with the user's explicit approval.

HTTP status: the intermittent in-process MOVE timeout has not reproduced with
thread dumps enabled. The five-second HTTP test timeout is unchanged. The new
stress coverage and diagnostic capture remain enabled; this issue is not
declared root-caused or resolved merely because local reruns passed.

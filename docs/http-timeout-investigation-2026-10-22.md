# Intermittent HTTP timeout investigation

Baseline: `07646da`, Windows host, Python 3.12.10. This investigation adds
diagnostics and regression coverage, not a runtime fix or timeout increase.

## Reusable capture tooling

The validated custom profile and capture helper are now maintained as
`scripts/http-storage.wprp` and `scripts/capture-http.ps1`. Run the latter in
administrator PowerShell, optionally with `-Rounds 6`. It stores private ETL,
test reports, and isolated pytest temporary directories beneath `.tmp_tests`;
these generated artifacts must not be committed. Historical `.tmp_tests`
script paths below describe the local investigation prototypes.

## Reproduced failure and full traces

The failure recurred on 2026-10-22 at about 23:18 Pacific in three independent
pytest workers: two directory MOVEs and a range GET. The original five-second
HTTP deadline remains unchanged. Each request has a one-second diagnostic
watchdog, and a second snapshot is taken at timeout.

Full traces are retained outside test fixture cleanup in `.tmp_tests/http-timeouts/`:

| Early snapshot | Request | Frozen request-thread location |
| --- | --- | --- |
| `timeout-66cf6c8d90cd44099a0973248e9e84be.txt` | MOVE | `ntpath.realpath`, via WsgiDAV containment checking |
| `timeout-6c48f6689ff642b99e9029b11ad27536.txt` | MOVE | `shutil.move` calling `os.rename` |
| `timeout-4d567cc592d342e8900a13cca9800ab8.txt` | range GET | `mimetypes.init` checking `os.path.isfile` |

Corresponding five-second snapshots are `timeout-14db19c7e8e54c859f0530f5cb8aca23.txt`,
`timeout-593715b3e97e4e128e44f6aed5a03845.txt`, and
`timeout-29b8d55c73c44431a9c7aedc6e5ce3e1.txt`. The complete pytest failure report
is `.tmp_tests/http-investigation-v5.xml`. These local diagnostic artifacts are
ignored by Git, not silently omitted from the investigation.

## Evidence and limits

- Request threads remain at the same filesystem calls in both snapshots.
- Independent in-memory heartbeat threads continued running: maximum gaps were
  0.063 or 0.079 seconds. This rules out a multi-second whole-process/GIL pause
  for these failures.
- Early snapshots were captured around monotonic time 264462.5 but only saved
  at 264480.6, about 18 seconds later. Even creating/writing the diagnostic files
  was delayed. Timeout snapshots took another 14 seconds to save.
- Scheduler/writeback workers were idle, not in a circular lock wait.
- Earlier diagnostics kept live frame references while formatting them. Threads
  could advance before line numbers were recorded. The replacement freezes
  stacks before any filesystem access and records capture/save times separately.
- The host reports its C: NVMe drive online/healthy. This is not proof against
  intermittent storage or filesystem-filter latency. No relevant sleep/wake
  event or System error was found for the sampled incident window.

The immediate cause of the HTTP timeout is a host filesystem stall exceeding
the client deadline, not a scheduler deadlock or merely slow simulated HDD
timing. The lower-level cause (storage, a filter driver, or another host
component) is **not yet identified**. Removing containment checks, skipping
durability flushes, retrying non-idempotent MOVE, or raising test timeouts would
not establish a safe fix.

## Experiments

- Repeated short-lived COPY/MOVE workflows at 4 and 16 workers.
- Long-lived server: 100 COPY/MOVE/GET/DELETE cycles with 10 ms dirty expiry to
  exercise background writeback.
- Concurrent audio rendering and the full Python suite; the mixed full run had
  one separate performance-expectation failure that passed in isolation.
- Restart after a one-minute idle interval.
- Healthy repetitions and cross-platform CI from the baseline are not treated
  as evidence that this intermittent host stall is resolved.

Next discriminating step: a short, scoped Windows filesystem/storage trace
during reproduction. Windows Performance Recorder is present. Such recording
needs elevated host tracing and may include paths/processes outside this repo;
obtain user approval before collecting it. Do not disable security software or
change storage settings as an unverified workaround.

## Storage trace permission blocker

The user approved a short elevated storage trace. WPR reported no existing
recording before the attempt. Starting the built-in DiskIO, FileIO, and
Minifilter profiles in default memory mode failed with error `0xc5585011`:
"Failed to enable the policy to profile system performance."
This occurred even through the execution tool's escalation path. A subsequent
`wpr -status` confirmed that no recording was running. No ETL was collected,
and no tracing session was cancelled. An administrator PowerShell session is
needed to proceed; the lower-level cause remains unresolved.

## Administrator recording and follow-up

The user started and saved an administrator recording locally at
`.tmp_tests/http-storage-20261022.etl` (3,810,525,184 bytes). The recording is
private diagnostic data and must not be committed or uploaded. ETW collectors
were visible through `logman query -ets`, although this execution session's
WPR status/stop commands could not access the administrator recording.

The traced test run began at 2026-10-22 23:36:09 Pacific and finished in 19.74
seconds: 48 passed, two failed. Directory population returned HTTP 500 on PUT;
the long-lived workflow returned HTTP 403 on DELETE. Neither failure was an
HTTP timeout, and no new watchdog stack snapshot was produced. The original
test helper discarded HTTP error bodies, and WsgiDAV's `verbose: 0` configured
its logger at CRITICAL, suppressing useful server exception reports. Those
missing details cannot be reconstructed from the JUnit report.

The helper now leaves logging configuration to pytest and prints bounded HTTP
error bodies. With those diagnostics enabled, the 50-test HTTP suite passed,
both previously failing tests passed in isolation, and ten further HTTP suite
runs passed (500 tests). The full Python suite passed (283 tests in 43.57 s),
and lint/type checks passed. These results do not establish that either the
host stall or the new HTTP failures has been fixed. No production behavior or
timeouts have been changed.

### Retained trace coverage

`tracerpt` processed 26,577,911 events over 190 seconds and reported zero lost
events. However, the earliest retained event for the classic FileIo provider
(`90cbdc39-4a3e-11d1-84f4-0000f80464e3`), queried chronologically using
`Get-WinEvent -Oldest`, is at 23:38:00 Pacific. This is after both failing
requests. Storage-provider events exist at 23:36:09, but the corresponding
file-operation history needed for attribution is absent. Zero lost events
does not establish complete time coverage of a circular memory recording.
The gap is consistent with memory-buffer overwrite before the manual save.
It does not implicate any particular filter driver or establish a disk fault.

Prepared `.tmp_tests/capture-http.ps1` as a local administrator-only diagnostic
helper. It starts file-mode DiskIO/FileIO/Minifilter tracing, runs up to three
HTTP suite repetitions by default, stops on the first failure, and saves the
trace in a finally block in the same elevated session. Each run uses a unique
ignored output directory. It never cancels another recording. Syntax parsing
passed; actual elevated capture still requires the user's administrator
session. No raw trace data has been uploaded or committed.

### File-mode capture at 2026-10-23 00:00 Pacific

The user's administrator run saved output in
`.tmp_tests/http-capture-9e6f8e04de1d4d988b0ce962b80a1a0b`.
All three JUnit reports confirm 50 passes each, no errors or failures, in
18.078, 16.022, and 19.203 seconds. The ETL is 7,969,177,600 bytes. WPR reported
318,897,149 dropped events. There is no reproduced failure in this run and
the incomplete trace cannot rule out a storage/filter issue. Do not repeat
the same high-volume capture configuration or infer a runtime fix.

The local capture helper now defaults to a single repetition and uses
`DiskIO.light` plus `FileIO.light`, without Minifilter. The installed WPR
accepted these combined file-mode profile details. Microsoft documents lower
overhead for light recordings:
https://learn.microsoft.com/en-us/windows-hardware/test/wpt/introduction-to-wpr
The helper also retains WPR stop output and returns a nonzero result for a
save failure or the reported English dropped-events warning. PowerShell
syntax and positive/zero event-loss recognition checks passed. Actual trace
volume and completeness of this reduced configuration remain unverified and
require a single administrator pilot run, not another prolonged capture.

### Reduced capture at 2026-10-23 00:06 Pacific

The reduced-profile pilot saved output in
`.tmp_tests/http-capture-68c0b20168d34b1780090babad317093`.
All 50 tests passed, but WPR still reported 73,224 dropped events. The saved
ETL is 634,388,480 bytes. The helper correctly retained the warning in
`wpr-stop.log` and flagged the capture as incomplete. Lower volume is not
proof of complete evidence or resolution. There was no reproduced timeout.

Two file-mode configurations have now lost events. Stop asking for repeated
administrator captures with these profiles. Preserve the existing private
recordings, but prioritize process-local exception/stack diagnostics for the
next reproduction. Do not disable filters, weaken durability or containment,
or lengthen the HTTP deadline as an unverified fix. The root cause remains
unresolved; no runtime fix has been committed or pushed.

Inspection of existing watchdog snapshots also found four one-second alerts
from the earlier heavy file-mode recording at 00:01:28 (not the reduced run).
Snapshots `8289c3dc058a4fe9a4ebe2033a4b5356`,
`bf1c8b0f8bab4674b27245099ee0e032`,
`c3f6b33dfca4490996651c26ebaf89e1`, and
`d0f4915fd04c4671aeb4d2e3dd6b2597` under `.tmp_tests/http-timeouts`
show MOVE/PUT/COPY requests crossing the watchdog threshold together. Frozen
server stacks show three temporary-file closes during persistence and one
`os.rename` inside `shutil.move`. Heartbeat maximum gaps were 63 ms; snapshots
were saved 0.17-0.28 seconds after capture. All requests ultimately passed.
These observations support filesystem-call delays, not a process-wide pause,
but heavy tracing itself is a confounder. They do not identify the component
responsible for the original roughly 19-second stalls.

### Custom buffered pilot prepared

At the user's request, replaced the capture helper's built-in profiles with
`.tmp_tests/http-storage.wprp!HttpStorage`. This defines one system collector
with 256 buffers of 1024 KB (256 MiB configured), and only DiskIO, DiskIOInit,
FileIO, FileIOInit, Loader, and ProcessThread keywords. It has no user event
collector, minifilter callbacks, CPU sampling, or stack walks. This removes
the unrelated providers from the built-in light profiles as well as increasing
the system collector's configured buffering from 40 MiB to 256 MiB.

The installed WPR accepted the custom file-mode profile and displayed the
expected buffer count and exact keyword set. The updated PowerShell script
passed syntax parsing and saves its profile and profile details with each run.
The default remains one test round, followed immediately by trace save. Actual
event loss and runtime overhead still require the administrator pilot; profile
validation is not a successful recording or a fix for the HTTP timeout.

### Custom profile pilot verified

The administrator pilot at 00:11 on October 23 saved
`.tmp_tests/http-capture-4bd86ede7d1d4395b2114adb54681780`.
JUnit confirms 50 passes, no errors/failures, in 12.293 seconds beginning
00:11:10.214556 Pacific. The earliest retained FileIo event is
00:11:09.316912, before the tests. No new one-second watchdog snapshots were
created. The ETL is 512,753,664 bytes. `tracerpt` processed 5,230,299 events
over 15 seconds and reported zero lost events. It also warned that some
events do not match the installed decoding schema; event-count completeness
is distinct from successful payload decoding.

This validates a short, loss-free baseline for the custom recording profile,
not a timeout fix or a guarantee that longer captures remain loss-free. The
next bounded reproduction can use the existing helper with `-Rounds 6`, which
stops on the first test failure and saves immediately. Expect several GB of
private trace output if all rounds pass; check event loss again before using
the resulting trace for causal claims.

### Six-round custom capture

At 00:14-00:15 Pacific the user captured six rounds in
`.tmp_tests/http-capture-29923d9899da4a0ab3efab8b2aa2120b`.
All six JUnit reports show 50 passes and zero errors/failures, for 300 passes
total. Durations were 12.113, 11.229, 12.065, 11.743, 12.377, and 13.509
seconds. No new one-second watchdog snapshots appeared. WPR's saved stop log
reports success without an event-loss warning. The ETL is 2,218,786,816 bytes;
its payload has not been fully decoded and counted. There is no failing or
watchdog-triggering request in this capture to attribute to a component.

Execution context is a confounder: the helper runs pytest in the user's
administrator session, while the agent's original execution context is not
administrator (confirmed by WindowsPrincipal.IsInRole even via the tool's
escalation path). This does not prove permissions caused the failures. Compare
the original context before asking for more identical administrator captures.

The original-context comparison completed four rounds of 50 passing tests
before the loop was interrupted during round five. Each completed round
reported an atexit PermissionError (WinError 5) when pytest inspected the
shared `pytest-of-yorke/pytest-current` temporary path. This is a separate
cleanup error, not an HTTP failure. Repeated cleanup errors are not a useful
reason to continue the same loop. The local capture helper now supplies a
unique `--basetemp` inside each capture output directory, isolating elevated
runs from pytest's shared per-user temporary root. No permissions or existing
temporary directories were changed or deleted. Script syntax passed; the
original HTTP timeout remains unexplained.

A subsequent original-context HTTP suite using a unique repository-local
`--basetemp` passed all 50 tests in 12.00 seconds and exited cleanly without
the atexit permission error. This validates the isolated temporary-directory
approach for that cleanup issue, not a production HTTP fix.

# Audit TODO

This file is the output of a line-by-line audit pass over the current codebase. Items are grouped by whether they were fixed during the pass or are still open follow-up work.

## Fixed In This Pass

- [x] Clamp filesystem reads to the inode size so the simulator does not price data past EOF inside a partially used final block.
- [x] Make startup stage partitioning respect the configured total wake time instead of letting stage minimums overflow the requested budget.
- [x] Replace scheduler polling with proper queue wakeups so dispatch behaves like a queue instead of a busy loop.
- [x] Preserve partial-cache and startup metadata when aggregating multi-operation stats.
- [x] Keep asynchronous write stats honest when a journal write already caused real disk work.
- [x] Flush dirty write-back state during `reset_runtime_state()` so profiling resets are actually clean.
- [x] Track wrapped writer offsets after seeks instead of assuming every write stream stays at offset `0`.
- [x] Isolate profiling runs in temporary backing directories so benchmarks do not reuse repo leftovers.
- [x] Make profiling futures surface exceptions instead of silently swallowing worker failures.
- [x] Make the manual HTTP stress scripts fail fast on HTTP errors instead of continuing with bad responses.
- [x] Tie the generated spin-up sample to the runtime startup model so sample audio and live behavior describe the same machine state sequence.
- [x] Allow audio sample rendering to produce valid output even for empty or very short scenarios.

## Remaining Follow-Up Work

### Correctness

- [ ] Add explicit rename and move modeling in the virtual filesystem instead of treating the project as mostly create/read/write/delete.
- [ ] Add directory create/remove semantics and tests for nested path trees.
- [ ] Model partial-block overwrites more explicitly if the project ever needs sub-block read-modify-write behavior instead of whole-block simplification.
- [ ] Decide whether `VirtualHDD.access_file()` should synthesize lookup metadata automatically for direct callers, or stay as a lower-level data path helper.

### WebDAV Integration

- [ ] Intercept directory enumeration paths so `PROPFIND` / readdir-style traffic pays the same metadata model as file opens.
- [ ] Add explicit tests for WebDAV overwrite, delete, and directory listing behavior against a live local server.
- [ ] Check whether all wrapped file objects need explicit context-manager methods in addition to attribute proxying.

### HDD Model Realism

- [ ] Separate command overhead for metadata, journal, cached data, and flush operations instead of one shared constant.
- [ ] Decide whether the idle timers should be measured from last access or from entry into each lower-power state, and document that choice clearly.
- [ ] Add optional background maintenance tasks such as offline scan or thermal recalibration that compete with foreground I/O.
- [ ] Consider zone-boundary transfer handling for large requests that span more than one zone.
- [ ] Consider a profile system for different drive classes instead of one default 7200 RPM desktop-ish profile.

### Audio Model

- [ ] Add selectable installation profiles such as `bare_drive_lab`, `mounted_in_case`, and `external_enclosure`.
- [ ] Tie audio coloration more directly to seek class, write-back drain pressure, and head-load state instead of one default installation filter.
- [ ] Add regression tests for startup/park sample envelopes in addition to the existing seek spectral sanity check.

### Tooling

- [ ] Add a smoke test that boots the WebDAV server on an ephemeral port and exercises one end-to-end request.
- [ ] Add coverage for the profiling helpers so they are checked for obvious regressions even though they are not unit tests.
- [ ] Add a small `requirements.txt` or installation section that names the expected runtime packages explicitly.

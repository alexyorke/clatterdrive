# TODO

This file is intentionally pruned to unfinished work only. Completed items are removed instead of left behind as history.

## Highest Impact

### HDD Model Realism

- [ ] Add optional retry / ECC / read-recovery behavior so rare long-tail reads exist.
- [ ] Add optional thermal recalibration / background scan activity that competes with foreground I/O.
- [ ] Add smarter writeback flush clustering so deferred writes drain as grouped media work.
- [ ] Decide whether partial-block writes should remain block-granular or model read-modify-write on the tail block.

### WebDAV / Filesystem Semantics

- [ ] Add explicit simulator support for `PROPPATCH` and realistic DAV property churn.
- [ ] Decide whether DAV lock operations should be modeled as metadata traffic or intentionally ignored.
- [ ] Add finer-grained modeling for collection overwrite-copy so destination cleanup is represented explicitly if warranted.
- [ ] Add explicit support for overwrite-via-copy with partial destination reuse instead of always delete-then-recreate semantics.

## Correctness

- [ ] Add tests for case-only `COPY` and `MOVE` paths on Windows-backed storage.
- [ ] Add tests for large directory listings so `readdir` realism is validated beyond tiny trees.
- [ ] Add tests for repeated `PROPFIND` traffic from real DAV-style clients to validate lookup/listing cache behavior.
- [ ] Add tests for deleting or moving a directory while writeback is still queued for descendants.
- [ ] Add tests for deleting a file immediately after a large buffered write before background flush completes.
- [ ] Add tests for copy-overwrite of large files across multiple chunks while background writeback is active.
- [ ] Add tests for deep nested rename followed by `PROPFIND` and `GET`.
- [ ] Decide how much out-of-band backing-store mutation should be tolerated and document the resulting consistency model more precisely.
- [ ] Add explicit detection or reconciliation for files that change on disk without going through the simulator.

## Audio / Synthesis

- [ ] Remove any remaining implicit state coupling that makes sound depend on call timing more than emitted events.
- [ ] Add a first-class audio voice abstraction if the synth grows beyond the current spindle / actuator / cover structure.

## Nice To Have

- [ ] Add a debug trace view that prints emitted audio events separately from filesystem and latency logs.
- [ ] Add a simple visualization of power state, queue depth, and head position over time.
- [ ] Add a silent render / offline tracing mode that writes event streams to JSON.
- [ ] Add more sample scenarios such as copy-heavy workloads, idle-to-standby wake, and metadata storms.

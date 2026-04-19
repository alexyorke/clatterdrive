# TODO

This file is intentionally pruned to unfinished work only. Completed items are removed instead of left behind as history.

## Highest Impact

### HDD Model Realism

### Mechanical / Acoustic Engine

- [ ] Replace the remaining heuristic seek-target generation with a more explicit seek planner that uses velocity / acceleration / jerk limits.
- [ ] Move windage and bearing generation fully to plant-driven disturbance channels and remove any remaining dependence on host op-kind heuristics.
- [ ] Split airborne radiation and structure-borne transfer calibration more cleanly so acoustic profiles are output calibration only.
- [ ] Decide whether to introduce a dedicated `audio/plant.py` module if `audio/core.py` grows again.

### WebDAV / Filesystem Semantics

- [ ] Add explicit simulator support for `PROPPATCH` and realistic DAV property churn.
- [ ] Decide whether DAV lock operations should be modeled as metadata traffic or intentionally ignored.
- [ ] Add finer-grained modeling for collection overwrite-copy so destination cleanup is represented explicitly if warranted.
- [ ] Add explicit support for overwrite-via-copy with partial destination reuse instead of always delete-then-recreate semantics.

## Correctness

## Audio / Synthesis

- [ ] Remove any remaining implicit state coupling that makes sound depend on call timing more than emitted events.
- [ ] Add a first-class audio voice abstraction if the synth grows beyond the current spindle / actuator / cover structure.

## Nice To Have

- [ ] Add a debug trace view that prints emitted audio events separately from filesystem and latency logs.
- [ ] Add a simple visualization of power state, queue depth, and head position over time.
- [ ] Add a silent render / offline tracing mode that writes event streams to JSON.
- [ ] Add more sample scenarios such as copy-heavy workloads, idle-to-standby wake, and metadata storms.

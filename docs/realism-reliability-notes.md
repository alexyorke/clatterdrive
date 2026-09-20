# Mechanics and reliability review, 2026-09-19

This pass corrects specific model assumptions and completion semantics. It does
not establish measured acoustic accuracy for an individual drive.

## Physical evidence and implementation

- Seagate describes spindle whir, soft clicks during head movement, harder
  parking clicks, and amplification by the mounting surface. These support
  separate spindle, actuator, parking, and chassis paths already in the model.
  [Seagate sound guide](https://www.seagate.com/support/kb/identifying-hard-drive-sounds-and-determining-what-they-mean/)
- Healthy heads fly above the platter. Repeated reads, writes, and metadata
  operations no longer inject a head/media contact impulse. The load impulse
  represents ramp/suspension release; parking still excites the stop. Workload
  effects on servo reaction are normalized approximations, not measured forces.
  [Virginia Tech disk-drive explanation](https://opendsa-server.cs.vt.edu/ODSA/Books/fu/comp502/fall-2020/Franklin_Fall_2020/html/Diskdrive.html)
- Sequential track-change spacing now uses `60 / RPM / transfer_duty` seconds.
  At full duty, 7200 RPM gives 8.333 ms and 5400 RPM gives 11.111 ms. This is
  an inference from one track passing under a head per revolution. It does not
  reproduce vendor head-switch skew, track skew, partial tracks, or firmware
  prefetch. The old fixed cadence could not distinguish spindle speeds.

## Completion and persistence

- The synchronous raw block adapter serializes simulation and backing-store
  completion together. Flush, reads, discard, and close cannot overtake an
  active backing write. This sacrifices adapter concurrency until store updates
  can execute inside scheduler dispatch callbacks. WebDAV NCQ is unchanged.
- Short backing reads fail explicitly. Close flushes the store and rejects
  subsequent I/O; a failed close flush can be retried.
- Sidecar saves use unique temporary siblings, flush Python buffers, call
  `os.fsync`, then replace the destination. Write/flush/replace failures clean
  up the temporary file and propagate to the caller. A failed pre-replacement
  flush leaves the previous sidecar intact and the simulator dirty for retry.
  [Python fsync documentation](https://docs.python.org/3/library/os.html#os.fsync)
- Atomic replacement and flushing the temporary file do not make allocation
  state and host file contents one crash-consistent transaction. Directory
  entry durability after a power loss is still platform/filesystem dependent.
  Multiple simulator processes sharing one volume are not coordinated.
  [Microsoft file caching](https://learn.microsoft.com/en-us/windows/win32/fileio/file-caching)

## Validation

Regression tests cover healthy I/O without contact impulses, RPM-dependent
track cadence, failed sidecar flush recovery, blocked-write/flush ordering,
short reads, and operations after close. The existing audio reference bounds
and golden tolerances remain unchanged.

Sources were consulted on 2026-09-19. Public source links are retained here;
no recordings or downloaded reference bundles were added.

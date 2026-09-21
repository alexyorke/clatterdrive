# Real-time audio performance

Audio correctness and real-time throughput are separate requirements. At
44,100 Hz, each 1,024-frame buffer has a 23.22 ms playback budget. The earlier
sample-by-sample NumPy renderer missed that budget on the development machine.

## Implementation

- Motor and servo state still advance at every audio sample. At exact spindle
  equilibrium the motor/drag cancellation is evaluated directly; there is no
  epsilon snapping or reduced physics update rate.
- Resonators run as independent compiled second-order filters. For state
  matrix A their denominator is `[1, -trace(A), det(A)]`; initial delays encode
  the previous physical displacement and velocity. The scalar diagnostic
  renderer remains the reference implementation.
- Noise filtering, rotor harmonics, source routing, radiation and final output
  filtering operate on blocks. Startup transitions inside a block split the
  noise filter processing at the exact transition sample and retain its state.
- The live device callback only copies pre-rendered samples. Synthesis and WAV
  writes run on a producer thread with four queued chunks (roughly 0.1 seconds
  of lookahead, plus device latency). There is no process priority change.
- `playback_health()` counts empty-buffer callbacks, device-reported underflows,
  and producer errors. Producer failure yields silence instead of doing expensive
  recovery on the callback thread.

The filter state conversion follows [SciPy's documented difference equations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html).
Keeping expensive processing and file I/O out of the callback follows the
[sounddevice callback constraints](https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html).

## Checks

`tests/test_audio_performance.py` compares fast output with the scalar renderer
at 22.05, 44.1 and 48 kHz, across drive profiles, irregular chunks, seeks,
parking, and a startup transition inside a block. Absolute waveform tolerance
is 1e-10 with relative tolerance 1e-8. Tests also cover partial-buffer copying,
underrun silence, bounded lookahead, and worker cleanup after device failure.
The existing audio golden and startup acceptance bounds are unchanged.

Run this independently of other heavy tests or simulator instances:

```sh
uv run python -m tools.audio_realtime_benchmark
uv run python -m tools.audio_realtime_benchmark --live --seconds 30
```

The command reports median/p95/max chunk time and misses against the actual
buffer deadline. It fails if p95 misses the deadline, or if the optional live
test records an underrun or producer error. Results depend on hardware, output
device, workload and system load; numerical tests alone do not certify smooth
playback. The live check uses repeated seeks, not every possible workload.

On the Windows development host on 2026-09-20, the final 200-chunk run measured
5.90 ms median, 12.02 ms p95 and 19.86 ms maximum against 23.22 ms deadlines,
with zero missed chunks. A 30-second run on the configured headphone output
reported zero buffer underruns, zero device underflows and no producer error.
The earlier unoptimized renderer measured about 48 ms median under the then
current system load. These are host-specific observations, not CI speed limits.

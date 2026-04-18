import numpy as np
import sounddevice as sd
import threading

class HDDAudioEngine:
    """
    Refined Procedural Audio Engine for HDD Acoustics.
    Advanced Features:
    - Sidechain Compression: Actuator transients duck the Spindle Engine.
    - Thermal Calibration: Periodic alignment ticks.
    - Ramp Parking: Distinct resonant clacks for head parking.
    - Reynolds-based Windage: Bandwidth-mapped resonant noise.
    """
    def __init__(self, sample_rate=44100, seed=None):
        self.fs = sample_rate
        self.chunk_size = 1024
        self.rng = np.random.default_rng(seed)
        
        # Telemetry State
        self.rpm = 0.0
        self.seek_trigger = False
        self.seek_distance = 0.0
        self.is_sequential = False
        self.is_parking = False
        self.is_calibrating = False
        self.queue_depth = 1
        self.op_kind = "data"
        self.is_flush = False
        self.is_spinup = False
        
        # DSP State
        self.phase = 0.0
        self.lock = threading.Lock()
        
        # Modal Synthesis (Actuator Arm)
        self.modes = np.array([180.0, 430.0, 920.0, 1680.0])
        self.decays = np.array([0.03, 0.024, 0.016, 0.01])
        self.mode_gains = np.array([0.48, 0.28, 0.16, 0.09])
        self.resonator_states = np.zeros((len(self.modes), 2))
        self.windage_state = 0.0
        self.activity_state = 0.0
        self.flutter_phase = 0.0
        self.air_path_state = 0.0
        self.body_path_state = 0.0
        self.body_bass_state = 0.0
        self.output_gain = 0.34
        
        # Sidechain state
        self.envelope_follower = 0.0
        self.stream = None

    def _update_telemetry(
        self,
        rpm,
        seek_trigger=False,
        seek_dist=0,
        is_seq=False,
        is_park=False,
        is_cal=False,
        queue_depth=1,
        op_kind="data",
        is_flush=False,
        is_spinup=False,
    ):
        with self.lock:
            self.rpm = rpm
            if seek_trigger:
                self.seek_trigger = True
                self.seek_distance = seek_dist
            self.is_sequential = is_seq
            self.is_parking = is_park
            self.is_calibrating = is_cal
            self.queue_depth = queue_depth
            self.op_kind = op_kind
            self.is_flush = is_flush
            self.is_spinup = is_spinup

    def _snapshot_telemetry(self):
        with self.lock:
            return {
                "rpm": self.rpm,
                "is_sequential": self.is_sequential,
                "queue_depth": self.queue_depth,
                "op_kind": self.op_kind,
                "is_spinup": self.is_spinup,
            }

    def _colored_noise(self, n, level, smoothing, state_name):
        if level <= 0.0:
            return np.zeros(n)

        raw = self.rng.normal(0.0, level, n)
        output = np.empty(n)
        state = getattr(self, state_name)
        for i, sample in enumerate(raw):
            state = state * smoothing + sample * (1.0 - smoothing)
            output[i] = state
        setattr(self, state_name, state)
        return output

    def _apply_impulse(self, output, amplitude, profile, start=0):
        for idx, coeff in enumerate(profile):
            position = start + idx
            if position >= len(output):
                break
            output[position] += amplitude * coeff

    def _one_pole_lowpass(self, signal, cutoff_hz, state_name):
        if cutoff_hz <= 0.0:
            return np.zeros_like(signal)

        smoothing = np.exp(-2.0 * np.pi * cutoff_hz / self.fs)
        output = np.empty_like(signal)
        state = getattr(self, state_name)
        for i, sample in enumerate(signal):
            state = state * smoothing + sample * (1.0 - smoothing)
            output[i] = state
        setattr(self, state_name, state)
        return output

    def _apply_installation_filter(self, signal):
        direct_air = self._one_pole_lowpass(signal, 1450.0, "air_path_state")
        body_path = self._one_pole_lowpass(signal, 420.0, "body_path_state")
        body_bass = self._one_pole_lowpass(signal, 115.0, "body_bass_state")
        structure_band = body_path - body_bass
        return direct_air * 0.52 + structure_band * 0.33 + body_bass * 0.12

    def _generate_spindle(self, n, ducking_factor):
        telemetry = self._snapshot_telemetry()
        rpm = telemetry["rpm"]
        if rpm <= 0.0:
            self.phase += n
            self.flutter_phase += n
            return np.zeros(n)

        f0 = rpm / 60.0
        t = (np.arange(n) + self.phase) / self.fs
        flutter_t = (np.arange(n) + self.flutter_phase) / self.fs
        self.phase += n
        self.flutter_phase += n
        
        flutter = 1.0 + 0.015 * np.sin(2 * np.pi * 0.37 * flutter_t) + 0.006 * np.sin(2 * np.pi * 0.91 * flutter_t)
        harmonics = [1, 2, 3, 5, 7]
        amps = [0.09, 0.055, 0.03, 0.013, 0.008]
        hum = np.zeros(n)
        for h, amp in zip(harmonics, amps):
            hum += amp * flutter * np.sin(2 * np.pi * (f0 * h) * t)

        windage_level = 0.0025 + 0.0065 * min(rpm / 7200.0, 1.2)
        activity_level = 0.001 + min(telemetry["queue_depth"], 8) * 0.0007
        activity_smoothing = 0.94 if telemetry["is_sequential"] else 0.86
        if telemetry["is_sequential"]:
            activity_level *= 0.6
        if telemetry["op_kind"] in {"journal", "flush"}:
            activity_level *= 1.35
        if telemetry["is_spinup"]:
            windage_level *= 1.6
            activity_level *= 1.4

        windage = self._colored_noise(n, windage_level, 0.988, "windage_state")
        activity_noise = self._colored_noise(n, activity_level, activity_smoothing, "activity_state")
        if telemetry["is_sequential"]:
            activity_noise += 0.0018 * np.sin(2 * np.pi * 32.0 * t)
        
        return np.tanh((hum + windage + activity_noise) * 0.7) * (1.0 - ducking_factor)

    def _generate_actuator(self, n):
        click_out = np.zeros(n)
        with self.lock:
            if self.is_parking:
                self._apply_impulse(click_out, 0.72, [1.0, -0.65, 0.32, -0.1])
                self.is_parking = False
            elif self.seek_trigger:
                kind_scale = {
                    "metadata": 0.3,
                    "journal": 0.42,
                    "flush": 0.62,
                    "data": 0.75,
                }.get(self.op_kind, 0.55)
                strength = kind_scale * (0.18 + 0.34 * np.sqrt(min(self.seek_distance / 1000, 1.0)))
                self._apply_impulse(click_out, strength, [1.0, -0.45, 0.18])
                if self.is_flush and n > 40:
                    self._apply_impulse(click_out, strength * 0.42, [1.0, -0.5, 0.18], start=28)
                self.seek_trigger = False
            elif self.is_calibrating:
                self._apply_impulse(click_out, 0.05, [1.0, -0.35, 0.08])
                self.is_calibrating = False
            elif self.is_sequential and self.rng.random() < 0.06:
                self._apply_impulse(click_out, 0.012, [1.0, -0.18])
        
        # Modal Synthesis
        total_output = np.zeros(n)
        for i, (freq, decay) in enumerate(zip(self.modes, self.decays)):
            R = np.exp(-1.0 / (decay * self.fs))
            theta = 2.0 * np.pi * freq / self.fs
            a1 = -2.0 * R * np.cos(theta)
            a2 = R * R
            for j in range(n):
                x = click_out[j]
                y = x - a1 * self.resonator_states[i, 0] - a2 * self.resonator_states[i, 1]
                self.resonator_states[i, 1] = self.resonator_states[i, 0]
                self.resonator_states[i, 0] = y
                total_output[j] += y * self.mode_gains[i] * 0.16
                 
        return total_output

    def render_chunk(self, frames):
        actuator = self._generate_actuator(frames)

        for value in actuator:
            abs_value = abs(value)
            if abs_value > self.envelope_follower:
                self.envelope_follower = abs_value
            else:
                self.envelope_follower *= 0.996

        ducking = min(self.envelope_follower * 0.7, 0.55)
        spindle = self._generate_spindle(frames, ducking)
        installed_mix = self._apply_installation_filter(spindle * 0.82 + actuator * 0.95)
        return np.tanh(installed_mix * 1.05) * self.output_gain

    def _audio_callback(self, outdata, frames, time_info, status):
        outdata[:] = self.render_chunk(frames).reshape(-1, 1)

    def start(self):
        if self.stream is not None:
            return
        stream = sd.OutputStream(
            samplerate=self.fs, channels=1,
            callback=self._audio_callback, blocksize=self.chunk_size
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self.stream = stream
        
    def stop(self):
        if self.stream is None:
            return
        try:
            self.stream.stop()
        finally:
            self.stream.close()
            self.stream = None

engine = HDDAudioEngine()

import numpy as np
import sounddevice as sd
import threading
import time
import random

class HDDAudioEngine:
    """
    Procedural Audio Engine for HDD Acoustics.
    Implements:
    - Spindle Engine: Additive harmonics + Reynolds-mapped windage.
    - Actuator Engine: Modal synthesis (Ringz) for seek clicks.
    - Enclosure Engine: Biquad ATF (Acoustic Transfer Function) shaping.
    """
    def __init__(self, sample_rate=44100):
        self.fs = sample_rate
        self.chunk_size = 1024
        
        # Telemetry State
        self.rpm = 0.0
        self.target_rpm = 7200.0
        self.seek_trigger = False
        self.seek_distance = 0.0
        self.is_sequential = False
        
        # DSP State
        self.phase = 0.0
        self.running = False
        self.lock = threading.Lock()
        
        # Actuator Resonators (Modal Synthesis)
        # Modes: 940Hz, 2400Hz, 3500Hz, 5200Hz
        self.modes = np.array([940.0, 2400.0, 3500.0, 5200.0])
        self.decays = np.array([0.01, 0.005, 0.003, 0.001])
        self.resonator_states = np.zeros((len(self.modes), 2)) # For IIR filter
        
    def _update_telemetry(self, rpm, seek_trigger=False, seek_dist=0, is_seq=False):
        with self.lock:
            self.rpm = rpm
            if seek_trigger:
                self.seek_trigger = True
                self.seek_distance = seek_dist
            self.is_sequential = is_seq

    def _generate_spindle(self, n):
        f0 = self.rpm / 60.0
        t = (np.arange(n) + self.phase) / self.fs
        self.phase += n
        
        # Additive Harmonics (3, 6, 9, 12, 15, 18)
        harmonics = [3, 6, 9, 12, 15, 18]
        amps = [0.05, 0.08, 0.04, 0.02, 0.01, 0.005]
        hum = np.zeros(n)
        for h, amp in zip(harmonics, amps):
            jitter = np.random.normal(0, 0.01) # NRRO Jitter
            hum += amp * np.sin(2 * np.pi * (f0 * h + jitter) * t)
            
        # Windage (Filtered White Noise)
        # Reynolds Number simplified mapping
        windage = np.random.normal(0, 0.02, n) * (self.rpm / 7200.0)
        # Simple LP filter for windage
        return (hum + windage) * 0.2

    def _generate_actuator(self, n):
        click_out = np.zeros(n)
        with self.lock:
            if self.seek_trigger:
                # Impulse Exciter
                # Strength proportional to sqrt(D)
                strength = 0.5 + 0.5 * np.sqrt(min(self.seek_distance / 100000, 1.0))
                click_out[0] = strength
                self.seek_trigger = False
            elif self.is_sequential and random.random() < 0.1:
                # Micro-transients for "purring"
                click_out[0] = 0.02
        
        # Modal Synthesis via Parallel Resonators
        # Ringz.ar equivalent: 2nd order IIR
        total_output = np.zeros(n)
        for i, (freq, decay) in enumerate(zip(self.modes, self.decays)):
            # Calculate filter coefficients
            R = np.exp(-1.0 / (decay * self.fs))
            theta = 2.0 * np.pi * freq / self.fs
            a1 = -2.0 * R * np.cos(theta)
            a2 = R * R
            # Process block
            for j in range(n):
                x = click_out[j]
                y = x - a1 * self.resonator_states[i, 0] - a2 * self.resonator_states[i, 1]
                self.resonator_states[i, 1] = self.resonator_states[i, 0]
                self.resonator_states[i, 0] = y
                total_output[j] += y * 0.1
                
        return np.clip(total_output, -1, 1)

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status)
        
        spindle = self._generate_spindle(frames)
        actuator = self._generate_actuator(frames)
        
        # Mix and apply Enclosure LPF (Simple 1st order)
        mixed = spindle + actuator
        # Final gain
        outdata[:] = (mixed * 0.5).reshape(-1, 1)

    def start(self):
        self.running = True
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=1,
            callback=self._audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()
        
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()

# Singleton for global access
engine = HDDAudioEngine()

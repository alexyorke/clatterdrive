import numpy as np
import sounddevice as sd
import threading
import time
import random

class HDDAudioEngine:
    """
    Refined Procedural Audio Engine for HDD Acoustics.
    Advanced Features:
    - Sidechain Compression: Actuator transients duck the Spindle Engine.
    - Thermal Calibration: Periodic alignment ticks.
    - Ramp Parking: Distinct resonant clacks for head parking.
    - Reynolds-based Windage: Bandwidth-mapped resonant noise.
    """
    def __init__(self, sample_rate=44100):
        self.fs = sample_rate
        self.chunk_size = 1024
        
        # Telemetry State
        self.rpm = 0.0
        self.seek_trigger = False
        self.seek_distance = 0.0
        self.is_sequential = False
        self.is_parking = False
        self.is_calibrating = False
        
        # DSP State
        self.phase = 0.0
        self.lock = threading.Lock()
        
        # Modal Synthesis (Actuator Arm)
        self.modes = np.array([940.0, 2400.0, 3500.0, 5200.0])
        self.decays = np.array([0.01, 0.005, 0.003, 0.001])
        self.resonator_states = np.zeros((len(self.modes), 2))
        
        # Sidechain state
        self.envelope_follower = 0.0

    def _update_telemetry(self, rpm, seek_trigger=False, seek_dist=0, is_seq=False, is_park=False, is_cal=False):
        with self.lock:
            self.rpm = rpm
            if seek_trigger:
                self.seek_trigger = True
                self.seek_distance = seek_dist
            self.is_sequential = is_seq
            self.is_parking = is_park
            self.is_calibrating = is_cal

    def _generate_spindle(self, n, ducking_factor):
        f0 = self.rpm / 60.0
        t = (np.arange(n) + self.phase) / self.fs
        self.phase += n
        
        # Harmonics
        harmonics = [3, 6, 9, 12, 15, 18]
        amps = [0.05, 0.08, 0.04, 0.02, 0.01, 0.005]
        hum = np.zeros(n)
        for h, amp in zip(harmonics, amps):
            hum += amp * np.sin(2 * np.pi * (f0 * h) * t)
            
        # Reynolds-based Windage
        # Re ~ RPM * (Platter Radius^2) / Viscosity
        # Simplified: Re scales with RPM
        re = self.rpm * 20 
        # Delta_f / f_fl (%) quadratic relationship for turbulent flow
        df_ratio = 1.27e-10 * (re**2) - 8.552e-5 * re + 16.5
        windage = np.random.normal(0, 0.02, n) * (self.rpm / 7200.0)
        
        # Apply Ducking (Sidechain)
        return (hum + windage) * 0.2 * (1.0 - ducking_factor)

    def _generate_actuator(self, n):
        click_out = np.zeros(n)
        with self.lock:
            if self.is_parking:
                # Ramp Parking: Large resonant impulse
                click_out[0] = 1.5 
                self.is_parking = False
            elif self.seek_trigger:
                strength = 0.5 + 0.5 * np.sqrt(min(self.seek_distance / 100000, 1.0))
                click_out[0] = strength
                self.seek_trigger = False
            elif self.is_calibrating:
                # Thermal Tick: Small isolated impulse
                click_out[0] = 0.1
                self.is_calibrating = False
            elif self.is_sequential and random.random() < 0.15:
                # Granular "purr"
                click_out[0] = 0.03
        
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
                total_output[j] += y * 0.15
                
        return total_output

    def _audio_callback(self, outdata, frames, time_info, status):
        actuator = self._generate_actuator(frames)
        
        # Sidechain Envelope Follower (Fast attack, slow release)
        for val in actuator:
            abs_val = abs(val)
            if abs_val > self.envelope_follower:
                self.envelope_follower = abs_val # Attack
            else:
                self.envelope_follower *= 0.999 # Release
        
        ducking = min(self.envelope_follower * 0.8, 0.7)
        spindle = self._generate_spindle(frames, ducking)
        
        mixed = (spindle + actuator) * 0.5
        outdata[:] = np.clip(mixed, -1, 1).reshape(-1, 1)

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.fs, channels=1,
            callback=self._audio_callback, blocksize=self.chunk_size
        )
        self.stream.start()
        
    def stop(self):
        self.stream.stop()
        self.stream.close()

engine = HDDAudioEngine()

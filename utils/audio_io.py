"""Audio I/O utilities and test signal generation.

Handles all audio file reading/writing and provides deterministic test signal
generators (sine, noise, impulse, chirp) for plugin testing without external files.
All audio uses float32 numpy arrays in (channels, samples) layout — the format
both Pedalboard and DawDreamer expect.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import chirp as scipy_chirp


def load_audio(
    path: str,
    sample_rate: int | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Load an audio file and return (channels, samples) float32 array + sample rate.

    Args:
        path: Path to the audio file (WAV, FLAC, OGG, MP3 via soundfile).
        sample_rate: If set, resample to this rate. None preserves native rate.
        mono: If True, downmix to mono.

    Returns:
        Tuple of (audio_array, sample_rate). Array shape is (channels, samples).
    """
    # soundfile returns (samples, channels) — we need to transpose
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.T  # Now (channels, samples)

    if mono and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)

    if sample_rate is not None and sample_rate != sr:
        # Simple resampling via librosa (lazy import to keep startup fast)
        import librosa
        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled_channels.append(
                librosa.resample(y=audio[ch], orig_sr=sr, target_sr=sample_rate)
            )
        audio = np.stack(resampled_channels)
        sr = sample_rate

    return audio.astype(np.float32), sr


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int,
    bit_depth: int = 32,
) -> str:
    """Save a numpy array to a WAV file.

    Args:
        audio: Array shaped (channels, samples) in float32.
        path: Output file path. Parent directories are created automatically.
        sample_rate: Sample rate in Hz.
        bit_depth: 16, 24, or 32 (float).

    Returns:
        The absolute path to the written file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
    subtype = subtype_map.get(bit_depth, "FLOAT")

    # soundfile expects (samples, channels)
    sf.write(path, audio.T, sample_rate, subtype=subtype)
    return str(Path(path).resolve())


def audio_stats(audio: np.ndarray) -> dict:
    """Compute basic statistics for an audio buffer.

    Returns dict with peak_db, rms_db, duration_samples, channels, and is_silent.
    """
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    return {
        "peak_linear": round(peak, 6),
        "peak_db": round(20 * np.log10(max(peak, 1e-10)), 2),
        "rms_linear": round(rms, 6),
        "rms_db": round(20 * np.log10(max(rms, 1e-10)), 2),
        "channels": audio.shape[0],
        "samples": audio.shape[1],
        "is_silent": peak < 1e-6,
    }


# ===========================================================================
# Test Signal Generators
# ===========================================================================
# All generators return (channels, samples) float32 arrays in [-1, 1] range.

def generate_sine(
    freq: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    channels: int = 1,
) -> np.ndarray:
    """Generate a pure sine wave."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono] * channels)


def generate_white_noise(
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    channels: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Generate white noise. Set seed for deterministic output."""
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration)
    return (rng.uniform(-amplitude, amplitude, (channels, n))).astype(np.float32)


def generate_pink_noise(
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate pink noise (1/f spectrum) via spectral shaping."""
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration)
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    freqs[0] = 1.0  # Avoid division by zero at DC
    X_pink = X / np.sqrt(freqs)
    pink = np.fft.irfft(X_pink, n=n)
    pink = (pink / np.max(np.abs(pink)) * amplitude).astype(np.float32)
    return pink.reshape(1, -1)


def generate_impulse(
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
) -> np.ndarray:
    """Generate a unit impulse (Dirac delta) at sample 0."""
    n = int(sample_rate * duration)
    signal = np.zeros((channels, n), dtype=np.float32)
    signal[:, 0] = 1.0
    return signal


def generate_silence(
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
) -> np.ndarray:
    """Generate silence."""
    n = int(sample_rate * duration)
    return np.zeros((channels, n), dtype=np.float32)


def generate_chirp(
    f0: float = 20.0,
    f1: float = 20000.0,
    duration: float = 5.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    method: Literal["logarithmic", "linear"] = "logarithmic",
) -> np.ndarray:
    """Generate a frequency sweep (chirp). Logarithmic gives equal time per octave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = scipy_chirp(t, f0=f0, f1=f1, t1=duration, method=method, phi=-90)
    return (signal * amplitude).astype(np.float32).reshape(1, -1)


def generate_test_signal(
    signal_type: str,
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 2,
    **kwargs,
) -> np.ndarray:
    """Dispatch to the appropriate test signal generator.

    Args:
        signal_type: One of 'sine', 'white_noise', 'pink_noise', 'impulse',
                     'silence', 'chirp'.
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of output channels.
        **kwargs: Passed to the specific generator (e.g., freq for sine).

    Returns:
        Audio array shaped (channels, samples).
    """
    generators = {
        "sine": generate_sine,
        "white_noise": generate_white_noise,
        "pink_noise": generate_pink_noise,
        "impulse": generate_impulse,
        "silence": generate_silence,
        "chirp": generate_chirp,
    }

    gen = generators.get(signal_type)
    if gen is None:
        raise ValueError(
            f"Unknown signal type '{signal_type}'. "
            f"Available: {list(generators.keys())}"
        )

    # Build kwargs, adding channels where the generator supports it
    gen_kwargs = {"duration": duration, "sample_rate": sample_rate, **kwargs}
    import inspect
    sig = inspect.signature(gen)
    if "channels" in sig.parameters:
        gen_kwargs["channels"] = channels

    audio = gen(**gen_kwargs)

    # Ensure correct channel count (some generators return mono)
    if audio.shape[0] < channels:
        audio = np.repeat(audio, channels, axis=0)

    return audio

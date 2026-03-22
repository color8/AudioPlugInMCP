"""Analysis engine — audio feature extraction, comparison, and measurement.

Wraps librosa for spectral analysis/feature extraction and pyloudnorm for
LUFS loudness measurement. Provides structured analysis results that Claude
can interpret to make decisions about plugin quality and sound character.
"""

from __future__ import annotations

import numpy as np


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int,
    analysis_types: list[str] | None = None,
) -> dict:
    """Run one or more analysis types on an audio buffer.

    Args:
        audio: Audio array shaped (channels, samples) as float32.
        sample_rate: Sample rate in Hz.
        analysis_types: List of analysis types to run. If None, runs all.
            Options: 'spectrum', 'loudness', 'dynamics', 'pitch', 'rhythm',
                     'features', 'quality'.

    Returns:
        Dictionary keyed by analysis type with structured results.
    """
    import librosa

    # Work with mono for most analyses
    if audio.ndim == 2 and audio.shape[0] > 1:
        mono = np.mean(audio, axis=0)
    elif audio.ndim == 2:
        mono = audio[0]
    else:
        mono = audio

    if analysis_types is None:
        analysis_types = ["spectrum", "loudness", "dynamics", "features"]

    results = {}

    if "spectrum" in analysis_types:
        results["spectrum"] = _analyze_spectrum(mono, sample_rate)

    if "loudness" in analysis_types:
        results["loudness"] = _analyze_loudness(audio, sample_rate)

    if "dynamics" in analysis_types:
        results["dynamics"] = _analyze_dynamics(mono, sample_rate)

    if "pitch" in analysis_types:
        results["pitch"] = _analyze_pitch(mono, sample_rate)

    if "rhythm" in analysis_types:
        results["rhythm"] = _analyze_rhythm(mono, sample_rate)

    if "features" in analysis_types:
        results["features"] = _analyze_features(mono, sample_rate)

    if "quality" in analysis_types:
        results["quality"] = _analyze_quality(audio, sample_rate)

    return results


def compare_audio(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int,
    metrics: list[str] | None = None,
) -> dict:
    """Compare two audio buffers across multiple metrics.

    Args:
        audio_a: First audio buffer (channels, samples).
        audio_b: Second audio buffer (channels, samples).
        sample_rate: Sample rate in Hz.
        metrics: List of comparison metrics. If None, runs all.
            Options: 'spectral_similarity', 'loudness_difference',
                     'correlation', 'mse', 'frequency_response'.

    Returns:
        Dictionary keyed by metric name with comparison values.
    """
    # Ensure same length by truncating to shorter
    min_len = min(audio_a.shape[-1], audio_b.shape[-1])
    a = audio_a[..., :min_len]
    b = audio_b[..., :min_len]

    # Work with mono
    mono_a = np.mean(a, axis=0) if a.ndim == 2 else a
    mono_b = np.mean(b, axis=0) if b.ndim == 2 else b

    if metrics is None:
        metrics = ["spectral_similarity", "loudness_difference", "correlation", "mse"]

    results = {}

    if "spectral_similarity" in metrics:
        results["spectral_similarity"] = _spectral_similarity(mono_a, mono_b, sample_rate)

    if "loudness_difference" in metrics:
        results["loudness_difference"] = _loudness_difference(a, b, sample_rate)

    if "correlation" in metrics:
        results["correlation"] = _waveform_correlation(mono_a, mono_b)

    if "mse" in metrics:
        mse = float(np.mean((mono_a - mono_b) ** 2))
        results["mse"] = round(mse, 8)
        results["mse_db"] = round(10 * np.log10(max(mse, 1e-12)), 2)

    if "frequency_response" in metrics:
        results["frequency_response"] = _frequency_response_diff(mono_a, mono_b, sample_rate)

    return results


# ===========================================================================
# Internal Analysis Functions
# ===========================================================================

def _analyze_spectrum(mono: np.ndarray, sr: int) -> dict:
    """Spectral shape descriptors."""
    import librosa

    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.95)
    bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=mono)
    contrast = librosa.feature.spectral_contrast(y=mono, sr=sr)

    return {
        "centroid_hz": round(float(np.mean(centroid)), 1),
        "centroid_std_hz": round(float(np.std(centroid)), 1),
        "rolloff_95_hz": round(float(np.mean(rolloff)), 1),
        "bandwidth_hz": round(float(np.mean(bandwidth)), 1),
        "flatness": round(float(np.mean(flatness)), 6),
        "contrast_db": [round(float(c), 2) for c in np.mean(contrast, axis=1)],
        "interpretation": _interpret_spectrum(
            float(np.mean(centroid)),
            float(np.mean(rolloff)),
            float(np.mean(flatness)),
        ),
    }


def _interpret_spectrum(centroid: float, rolloff: float, flatness: float) -> str:
    """Human-readable interpretation of spectral features."""
    parts = []

    if centroid < 1000:
        parts.append("dark/bass-heavy timbre")
    elif centroid < 3000:
        parts.append("warm/mid-focused timbre")
    elif centroid < 6000:
        parts.append("present/bright timbre")
    else:
        parts.append("very bright/airy timbre")

    if flatness > 0.3:
        parts.append("noise-like (non-tonal)")
    elif flatness > 0.1:
        parts.append("somewhat noisy")
    else:
        parts.append("tonal/harmonic")

    if rolloff < 5000:
        parts.append("bandwidth-limited (low-fi)")
    elif rolloff > 15000:
        parts.append("full-bandwidth")

    return "; ".join(parts)


def _analyze_loudness(audio: np.ndarray, sr: int) -> dict:
    """Loudness measurement including LUFS."""
    import pyloudnorm as pyln

    # pyloudnorm expects (samples, channels) — transpose from our (channels, samples)
    if audio.ndim == 1:
        audio_for_pyln = audio.reshape(-1, 1)
    else:
        audio_for_pyln = audio.T

    meter = pyln.Meter(sr)

    try:
        lufs = float(meter.integrated_loudness(audio_for_pyln))
    except Exception:
        lufs = -100.0  # Signal too quiet for LUFS measurement

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    return {
        "lufs": round(lufs, 2),
        "peak_db": round(20 * np.log10(max(peak, 1e-10)), 2),
        "rms_db": round(20 * np.log10(max(rms, 1e-10)), 2),
        "crest_factor_db": round(20 * np.log10(max(peak, 1e-10)) - 20 * np.log10(max(rms, 1e-10)), 2),
        "is_clipping": peak > 1.0,
    }


def _analyze_dynamics(mono: np.ndarray, sr: int) -> dict:
    """Dynamic range and envelope analysis."""
    import librosa

    rms = librosa.feature.rms(y=mono)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    return {
        "dynamic_range_db": round(float(np.max(rms_db) - np.min(rms_db[rms_db > -96])), 2),
        "rms_mean_db": round(float(np.mean(rms_db)), 2),
        "rms_std_db": round(float(np.std(rms_db)), 2),
    }


def _analyze_pitch(mono: np.ndarray, sr: int) -> dict:
    """Pitch detection and harmonic content."""
    import librosa

    # Pitch tracking via pYIN
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y=mono, fmin=50, fmax=4000, sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            return {
                "detected_pitch_hz": round(float(np.median(valid_f0)), 1),
                "pitch_range_hz": [round(float(np.min(valid_f0)), 1),
                                   round(float(np.max(valid_f0)), 1)],
                "pitch_stability": round(1.0 - float(np.std(valid_f0) / np.mean(valid_f0)), 4),
                "voiced_ratio": round(float(np.mean(voiced_flag)), 3),
            }
    except Exception:
        pass

    return {"detected_pitch_hz": None, "note": "No pitched content detected"}


def _analyze_rhythm(mono: np.ndarray, sr: int) -> dict:
    """Beat detection and tempo estimation."""
    import librosa

    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr)
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)

    return {
        "estimated_bpm": round(float(np.atleast_1d(tempo)[0]), 1),
        "beat_count": int(len(beats)),
        "onset_density": round(float(np.mean(onset_env)), 4),
    }


def _analyze_features(mono: np.ndarray, sr: int) -> dict:
    """High-level feature extraction (MFCCs, chroma, ZCR)."""
    import librosa

    mfccs = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=mono)

    return {
        "mfcc_means": [round(float(m), 3) for m in np.mean(mfccs, axis=1)],
        "chroma_profile": [round(float(c), 3) for c in np.mean(chroma, axis=1)],
        "zero_crossing_rate": round(float(np.mean(zcr)), 6),
    }


def _analyze_quality(audio: np.ndarray, sr: int) -> dict:
    """Signal quality metrics: noise floor, DC offset, phase correlation."""
    # DC offset
    dc_offset = float(np.mean(audio))

    results = {
        "dc_offset": round(dc_offset, 6),
        "has_dc_offset": abs(dc_offset) > 0.001,
    }

    # Phase correlation (stereo only)
    if audio.ndim == 2 and audio.shape[0] >= 2:
        left, right = audio[0], audio[1]
        if np.std(left) > 0 and np.std(right) > 0:
            corr = float(np.corrcoef(left, right)[0, 1])
            results["phase_correlation"] = round(corr, 4)
            if corr > 0.95:
                results["stereo_note"] = "Nearly mono (>0.95 correlation)"
            elif corr < 0:
                results["stereo_note"] = "Phase issues detected (negative correlation)"
            else:
                results["stereo_note"] = "Healthy stereo image"

    return results


def _spectral_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Cosine similarity between average magnitude spectra."""
    spec_a = np.abs(np.fft.rfft(a))
    spec_b = np.abs(np.fft.rfft(b))

    # Normalize
    norm_a = np.linalg.norm(spec_a)
    norm_b = np.linalg.norm(spec_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    similarity = float(np.dot(spec_a, spec_b) / (norm_a * norm_b))
    return round(similarity, 6)


def _loudness_difference(a: np.ndarray, b: np.ndarray, sr: int) -> dict:
    """Difference in LUFS and peak levels."""
    loud_a = _analyze_loudness(a, sr)
    loud_b = _analyze_loudness(b, sr)

    return {
        "lufs_diff": round(loud_b["lufs"] - loud_a["lufs"], 2),
        "peak_db_diff": round(loud_b["peak_db"] - loud_a["peak_db"], 2),
        "rms_db_diff": round(loud_b["rms_db"] - loud_a["rms_db"], 2),
    }


def _waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two waveforms."""
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 6)


def _frequency_response_diff(a: np.ndarray, b: np.ndarray, sr: int) -> dict:
    """Estimate frequency response difference (input vs output)."""
    n = min(len(a), len(b))
    spec_a = np.abs(np.fft.rfft(a[:n]))
    spec_b = np.abs(np.fft.rfft(b[:n]))

    # Avoid division by zero
    safe_a = np.maximum(spec_a, 1e-10)
    ratio_db = 20 * np.log10(spec_b / safe_a)

    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Sample at standard octave bands
    bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    band_gains = {}
    for band in bands:
        if band > sr / 2:
            break
        idx = np.argmin(np.abs(freqs - band))
        # Average over a small neighborhood for stability
        start = max(0, idx - 5)
        end = min(len(ratio_db), idx + 5)
        band_gains[f"{band}Hz"] = round(float(np.mean(ratio_db[start:end])), 2)

    return {"band_gain_db": band_gains}

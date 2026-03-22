"""Audio Plugin MCP Server — Main Entry Point.

A local MCP server that lets Claude interact with VST3/AU audio plugins:
load plugins, inspect parameters, process audio, render MIDI, analyze
spectral content, compare outputs, and compile JUCE projects.

Usage (stdio transport for Claude Desktop / Claude Code):
    python server.py

Configuration:
    Edit config.yaml in the same directory to set plugin paths, sample rates, etc.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# MCP SDK
from mcp.server.fastmcp import Context, FastMCP

# Local engines and utilities
from engines.pedalboard_engine import PedalboardEngine
from engines.analysis_engine import analyze_audio, compare_audio
from utils.audio_io import (
    audio_stats, generate_test_signal, load_audio, save_audio,
)
from utils.midi_utils import (
    MidiNote, load_midi_file, make_chord, make_scale, notes_to_mido_messages,
)


# ===========================================================================
# Configuration
# ===========================================================================

def load_config() -> dict:
    """Load config.yaml from the server directory."""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG = load_config()
SAMPLE_RATE = CONFIG.get("audio", {}).get("sample_rate", 44100)
CHANNELS = CONFIG.get("audio", {}).get("channels", 2)
CACHE_DIR = Path(CONFIG.get("cache", {}).get("directory", "./audio_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str = "") -> str:
    """Generate a unique path in the audio cache directory."""
    if not name:
        name = f"{uuid.uuid4().hex[:12]}.wav"
    return str(CACHE_DIR / name)


# ===========================================================================
# Application State (persists across tool calls via lifespan)
# ===========================================================================

@dataclass
class AppState:
    """Shared state accessible by all tools through the MCP context."""
    engine: PedalboardEngine = field(default_factory=PedalboardEngine)
    config: dict = field(default_factory=dict)
    sample_rate: int = 44100
    channels: int = 2


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Initialize persistent state on server start, clean up on shutdown."""
    state = AppState(
        config=CONFIG,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
    )
    try:
        yield state
    finally:
        # Cleanup: unload all plugins
        for name in list(state.engine.loaded_plugins.keys()):
            state.engine.unload(name)


# ===========================================================================
# MCP Server Instance
# ===========================================================================

mcp = FastMCP("audio_plugin_mcp", lifespan=app_lifespan)


def _get_state(ctx: Context) -> AppState:
    """Extract the persistent AppState from the MCP context."""
    return ctx.request_context.lifespan_context


# ===========================================================================
# Tool 0: Health Check
# ===========================================================================

@mcp.tool(
    name="health_check",
    annotations={
        "title": "Health Check",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
def health_check(ctx: Context) -> str:
    """Check server status and verify all dependencies are available.

    Returns system info, loaded plugin count, dependency versions, and
    configured plugin search paths.
    """
    import platform
    state = _get_state(ctx)
    info = {
        "status": "ok",
        "server": "audio_plugin_mcp",
        "platform": platform.system(),
        "python": platform.python_version(),
        "sample_rate": state.sample_rate,
        "loaded_plugins": len(state.engine.loaded_plugins),
        "cache_dir": str(CACHE_DIR.resolve()),
    }

    # Check dependency versions
    deps = {}
    try:
        import pedalboard
        deps["pedalboard"] = getattr(pedalboard, "__version__", "installed")
    except ImportError:
        deps["pedalboard"] = "MISSING"
    try:
        import librosa
        deps["librosa"] = librosa.__version__
    except ImportError:
        deps["librosa"] = "MISSING"
    try:
        import pyloudnorm
        deps["pyloudnorm"] = "installed"
    except ImportError:
        deps["pyloudnorm"] = "MISSING (LUFS measurement unavailable)"
    try:
        import dawdreamer
        deps["dawdreamer"] = "installed"
    except ImportError:
        deps["dawdreamer"] = "not installed (advanced MIDI rendering unavailable)"
    try:
        import mido
        deps["mido"] = "installed"
    except ImportError:
        deps["mido"] = "MISSING"

    info["dependencies"] = deps
    info["plugin_search_paths"] = CONFIG.get("plugin_search_paths", [])

    return json.dumps(info, indent=2)


# ===========================================================================
# Tool 1: Load Plugin
# ===========================================================================

@mcp.tool(
    name="load_plugin",
    annotations={
        "title": "Load Audio Plugin",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
def load_plugin(
    path: str,
    name: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Load a VST3 or AU plugin from a file path.

    Args:
        path: Full file path to the .vst3 or .component plugin.
        name: Friendly name for referencing this plugin in other tools.
              Defaults to the filename without extension.

    Returns:
        Plugin metadata including type (effect/instrument), parameter count,
        and the first 20 parameter names.
    """
    state = _get_state(ctx)

    try:
        instance = state.engine.load(path=path, name=name)
    except Exception as e:
        return f"Error loading plugin: {e}"

    params = instance.get_parameters()
    param_preview = [p["name"] for p in params[:20]]
    more = f" (and {len(params) - 20} more)" if len(params) > 20 else ""

    result = {
        "name": instance.name,
        "path": instance.path,
        "type": "instrument" if instance.is_instrument else "effect",
        "parameter_count": len(params),
        "parameters_preview": param_preview,
        "more_params": more,
    }
    return json.dumps(result, indent=2)


# ===========================================================================
# Tool 2: List Plugin Parameters
# ===========================================================================

@mcp.tool(
    name="list_plugin_parameters",
    annotations={
        "title": "List Plugin Parameters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
def list_plugin_parameters(
    plugin_name: str,
    ctx: Context = None,
) -> str:
    """List all parameters of a loaded plugin with their current values and metadata.

    Args:
        plugin_name: The name of a previously loaded plugin.

    Returns:
        Complete parameter list with names, raw values (0-1), labels,
        and whether each parameter is discrete or boolean.
    """
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        loaded = [p["name"] for p in state.engine.list_loaded()]
        return f"Plugin '{plugin_name}' not loaded. Loaded: {loaded}"

    params = inst.get_parameters()
    return json.dumps({"plugin": plugin_name, "parameters": params}, indent=2)


# ===========================================================================
# Tool 3: Set Plugin Parameters
# ===========================================================================

@mcp.tool(
    name="set_plugin_parameters",
    annotations={
        "title": "Set Plugin Parameters",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
def set_plugin_parameters(
    plugin_name: str,
    parameters: str,
    raw: bool = True,
    ctx: Context = None,
) -> str:
    """Set one or more parameters on a loaded plugin.

    Args:
        plugin_name: The name of a previously loaded plugin.
        parameters: JSON string of parameter name-value pairs.
                    Example: '{"ratio": 0.5, "attack_ms": 0.3}'
        raw: If true, values are normalized 0.0-1.0. If false, real-world units.

    Returns:
        Confirmation with actual values after setting (may differ due to discrete steps).
    """
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return f"Plugin '{plugin_name}' not loaded."

    try:
        param_dict = json.loads(parameters)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    try:
        results = inst.set_parameters_bulk(param_dict, raw=raw)
    except Exception as e:
        return f"Error setting parameters: {e}"

    return json.dumps({"plugin": plugin_name, "results": results}, indent=2)


# ===========================================================================
# Tool 4: Render Audio (Effect Processing)
# ===========================================================================

@mcp.tool(
    name="render_audio",
    annotations={
        "title": "Render Audio Through Plugin",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    },
)
def render_audio(
    plugin_name: str,
    input_path: Optional[str] = None,
    input_signal: Optional[str] = None,
    duration: float = 3.0,
    output_name: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Process audio through a loaded effect plugin and save the result.

    Provide EITHER input_path (existing audio file) OR input_signal
    (generated test signal). If neither, uses a 1kHz sine wave.

    Args:
        plugin_name: The name of a loaded effect plugin.
        input_path: Path to an input audio file (WAV, FLAC, etc.).
        input_signal: Test signal spec as JSON string. Example:
                      '{"type": "sine", "freq": 440}' or
                      '{"type": "white_noise"}' or
                      '{"type": "chirp", "f0": 20, "f1": 20000}'
                      Available types: sine, white_noise, pink_noise,
                      impulse, silence, chirp.
        duration: Duration in seconds (only used for generated signals).
        output_name: Output filename. Auto-generated if not provided.

    Returns:
        Path to the output file and basic audio statistics.
    """
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return f"Plugin '{plugin_name}' not loaded."
    if inst.is_instrument:
        return f"'{plugin_name}' is an instrument. Use render_midi_through_plugin instead."

    sr = state.sample_rate

    # Get input audio
    if input_path:
        try:
            audio, sr = load_audio(input_path, sample_rate=sr)
        except Exception as e:
            return f"Error loading audio: {e}"
    elif input_signal:
        try:
            spec = json.loads(input_signal)
            sig_type = spec.pop("type", "sine")
            audio = generate_test_signal(
                signal_type=sig_type, duration=duration,
                sample_rate=sr, channels=state.channels, **spec,
            )
        except Exception as e:
            return f"Error generating signal: {e}"
    else:
        audio = generate_test_signal("sine", duration=duration, sample_rate=sr,
                                      channels=state.channels, freq=1000.0)

    # Process
    try:
        output = inst.process_audio(audio, sample_rate=sr)
    except Exception as e:
        return f"Error processing audio: {e}"

    # Save
    out_path = _cache_path(output_name or f"{plugin_name}_{uuid.uuid4().hex[:8]}.wav")
    save_audio(output, out_path, sr)

    stats = audio_stats(output)
    stats["output_path"] = out_path
    stats["duration_seconds"] = round(output.shape[1] / sr, 3)
    stats["sample_rate"] = sr

    return json.dumps(stats, indent=2)


# ===========================================================================
# Tool 5: Analyze Audio
# ===========================================================================

@mcp.tool(
    name="analyze_audio",
    annotations={
        "title": "Analyze Audio File",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
def analyze_audio_tool(
    audio_path: str,
    analysis_types: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Analyze an audio file for spectral content, loudness, dynamics, and more.

    Args:
        audio_path: Path to the audio file to analyze.
        analysis_types: Comma-separated list of analysis types.
                        Options: spectrum, loudness, dynamics, pitch, rhythm,
                        features, quality. Default: all basic types.

    Returns:
        Structured analysis results keyed by analysis type.
    """
    state = _get_state(ctx)
    try:
        audio, sr = load_audio(audio_path)
    except Exception as e:
        return f"Error loading audio: {e}"

    types = None
    if analysis_types:
        types = [t.strip() for t in analysis_types.split(",")]

    try:
        results = analyze_audio(audio, sr, analysis_types=types)
    except Exception as e:
        return f"Error analyzing audio: {e}"

    results["file"] = audio_path
    results["duration_seconds"] = round(audio.shape[1] / sr, 3)
    results["sample_rate"] = sr
    results["channels"] = audio.shape[0]

    return json.dumps(results, indent=2, default=str)


# ===========================================================================
# Tool 6: Compare Audio
# ===========================================================================

@mcp.tool(
    name="compare_audio",
    annotations={
        "title": "Compare Two Audio Files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
def compare_audio_tool(
    path_a: str,
    path_b: str,
    metrics: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Compare two audio files across spectral, loudness, and waveform metrics.

    Args:
        path_a: Path to the first audio file (reference).
        path_b: Path to the second audio file (comparison).
        metrics: Comma-separated metrics. Options: spectral_similarity,
                 loudness_difference, correlation, mse, frequency_response.
                 Default: all metrics.

    Returns:
        Comparison scores for each requested metric.
    """
    try:
        audio_a, sr_a = load_audio(path_a)
        audio_b, sr_b = load_audio(path_b, sample_rate=sr_a)
    except Exception as e:
        return f"Error loading audio files: {e}"

    metric_list = None
    if metrics:
        metric_list = [m.strip() for m in metrics.split(",")]

    try:
        results = compare_audio(audio_a, audio_b, sr_a, metrics=metric_list)
    except Exception as e:
        return f"Error comparing audio: {e}"

    results["file_a"] = path_a
    results["file_b"] = path_b

    return json.dumps(results, indent=2, default=str)


# ===========================================================================
# Tool 7: Render MIDI Through Plugin
# ===========================================================================

@mcp.tool(
    name="render_midi_through_plugin",
    annotations={
        "title": "Render MIDI Through Instrument Plugin",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    },
)
def render_midi_through_plugin(
    plugin_name: str,
    midi_source: str,
    duration: float = 5.0,
    output_name: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Render MIDI notes through a loaded instrument plugin (synth/sampler).

    Args:
        plugin_name: The name of a loaded instrument plugin.
        midi_source: EITHER a path to a .mid file, OR a JSON spec for notes.
                     JSON spec examples:
                     - Single note: '{"note": "C4", "velocity": 100, "duration": 2.0}'
                     - Multiple notes: '{"notes": [{"note": 60, "velocity": 100,
                       "start": 0, "duration": 1}, ...]}'
                     - Chord: '{"chord": "C4", "type": "major", "duration": 2.0}'
                     - Scale: '{"scale": "C4", "type": "minor", "note_duration": 0.3}'
        duration: Total output duration in seconds.
        output_name: Output filename. Auto-generated if not provided.

    Returns:
        Path to rendered audio file and basic statistics.
    """
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return f"Plugin '{plugin_name}' not loaded."
    if not inst.is_instrument:
        return f"'{plugin_name}' is an effect. Use render_audio instead."

    sr = state.sample_rate

    # Parse MIDI source
    try:
        notes = _parse_midi_source(midi_source)
    except Exception as e:
        return f"Error parsing MIDI source: {e}"

    # Convert to mido messages for Pedalboard
    messages = notes_to_mido_messages(notes)

    # Render
    try:
        audio = inst.render_midi(
            messages, duration=duration, sample_rate=sr,
            num_channels=state.channels,
        )
    except Exception as e:
        return f"Error rendering MIDI: {e}"

    # Ensure correct shape (pedalboard may return different layouts)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    # Save
    out_path = _cache_path(output_name or f"{plugin_name}_midi_{uuid.uuid4().hex[:8]}.wav")
    save_audio(audio, out_path, sr)

    stats = audio_stats(audio)
    stats["output_path"] = out_path
    stats["duration_seconds"] = round(audio.shape[-1] / sr, 3)
    stats["note_count"] = len(notes)
    stats["sample_rate"] = sr

    return json.dumps(stats, indent=2)


def _parse_midi_source(source: str) -> list[MidiNote]:
    """Parse a MIDI source string into a list of MidiNotes."""
    # Check if it's a file path
    if source.endswith(".mid") or source.endswith(".midi"):
        if Path(source).exists():
            return load_midi_file(source)
        raise FileNotFoundError(f"MIDI file not found: {source}")

    # Parse as JSON spec
    spec = json.loads(source)

    # Single note shorthand
    if "note" in spec and "notes" not in spec:
        note_val = spec["note"]
        if isinstance(note_val, str):
            from utils.midi_utils import note_name_to_midi
            note_val = note_name_to_midi(note_val)
        return [MidiNote(
            note=note_val,
            velocity=spec.get("velocity", 100),
            start_time=spec.get("start", 0.0),
            duration=spec.get("duration", 1.0),
        )]

    # Multiple notes
    if "notes" in spec:
        return [
            MidiNote(
                note=n["note"] if isinstance(n["note"], int) else
                    __import__("utils.midi_utils", fromlist=["note_name_to_midi"]).note_name_to_midi(n["note"]),
                velocity=n.get("velocity", 100),
                start_time=n.get("start", 0.0),
                duration=n.get("duration", 1.0),
            )
            for n in spec["notes"]
        ]

    # Chord shorthand
    if "chord" in spec:
        return make_chord(
            root=spec["chord"],
            chord_type=spec.get("type", "major"),
            velocity=spec.get("velocity", 100),
            start_time=spec.get("start", 0.0),
            duration=spec.get("duration", 2.0),
        )

    # Scale shorthand
    if "scale" in spec:
        return make_scale(
            root=spec["scale"],
            scale_type=spec.get("type", "major"),
            velocity=spec.get("velocity", 100),
            note_duration=spec.get("note_duration", 0.5),
            start_time=spec.get("start", 0.0),
        )

    raise ValueError("Unrecognized MIDI spec format. Provide 'note', 'notes', 'chord', or 'scale'.")


# ===========================================================================
# Tool 8: Generate Preset (Iterative Optimization)
# ===========================================================================

@mcp.tool(
    name="generate_preset",
    annotations={
        "title": "Generate Plugin Preset via Optimization",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    },
)
def generate_preset(
    plugin_name: str,
    target: str,
    iterations: int = 20,
    ctx: Context = None,
) -> str:
    """Generate a plugin preset by iteratively optimizing toward a target sound.

    Uses random search with refinement: generate random parameter sets,
    render audio, analyze spectral features, and keep the best match.

    Args:
        plugin_name: A loaded effect or instrument plugin.
        target: EITHER a path to a reference audio file to match,
                OR a JSON description of target features, e.g.:
                '{"spectral_centroid_hz": 2000, "peak_db": -6, "brightness": "warm"}'
        iterations: Number of random parameter sets to try (more = better but slower).

    Returns:
        The best parameter set found and its similarity score.
    """
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return f"Plugin '{plugin_name}' not loaded."

    sr = state.sample_rate

    # Parse target
    target_features = None
    if Path(target).exists():
        ref_audio, _ = load_audio(target, sample_rate=sr)
        target_features = analyze_audio(ref_audio, sr, ["spectrum", "loudness"])
    else:
        try:
            target_features = json.loads(target)
        except json.JSONDecodeError:
            return "Target must be an audio file path or a JSON feature spec."

    # Generate test input signal
    test_input = generate_test_signal("chirp", duration=2.0, sample_rate=sr,
                                       channels=state.channels)

    # Store original parameters
    original_params = {p["name"]: p["raw_value"] for p in inst.get_parameters()}

    best_score = -1.0
    best_params = original_params.copy()
    param_names = list(original_params.keys())
    rng = np.random.default_rng(42)

    for i in range(iterations):
        # Generate random parameter set (mix of random and mutations of best)
        if i == 0:
            trial_params = original_params.copy()
        elif i < iterations // 2:
            trial_params = {name: float(rng.uniform(0, 1)) for name in param_names}
        else:
            # Refine around best found so far
            trial_params = {
                name: float(np.clip(best_params[name] + rng.normal(0, 0.1), 0, 1))
                for name in param_names
            }

        # Apply parameters
        try:
            inst.set_parameters_bulk(trial_params, raw=True)
        except Exception:
            continue

        # Render and analyze
        try:
            if inst.is_effect:
                output = inst.process_audio(test_input, sample_rate=sr)
            else:
                test_notes = notes_to_mido_messages(make_chord("C4", "major", duration=2.0))
                output = inst.render_midi(test_notes, duration=2.5, sample_rate=sr)
                if output.ndim == 1:
                    output = output.reshape(1, -1)
        except Exception:
            continue

        # Score against target
        try:
            features = analyze_audio(output, sr, ["spectrum", "loudness"])
            score = _score_features(features, target_features)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_params = trial_params.copy()

    # Restore best parameters
    inst.set_parameters_bulk(best_params, raw=True)

    return json.dumps({
        "plugin": plugin_name,
        "iterations": iterations,
        "best_score": round(best_score, 4),
        "best_parameters": {k: round(v, 4) for k, v in best_params.items()},
    }, indent=2)


def _score_features(actual: dict, target: dict) -> float:
    """Score how well actual features match target features. Returns 0-1."""
    scores = []

    # Match spectral centroid if available
    a_centroid = actual.get("spectrum", {}).get("centroid_hz")
    t_centroid = target.get("spectrum", {}).get("centroid_hz") or target.get("spectral_centroid_hz")
    if a_centroid and t_centroid:
        # Log-scale distance (octave-based)
        dist = abs(np.log2(max(a_centroid, 20) / max(t_centroid, 20)))
        scores.append(max(0, 1.0 - dist / 4.0))

    # Match loudness
    a_lufs = actual.get("loudness", {}).get("lufs")
    t_lufs = target.get("loudness", {}).get("lufs") or target.get("lufs")
    if a_lufs and t_lufs and a_lufs > -100 and t_lufs > -100:
        dist = abs(a_lufs - t_lufs)
        scores.append(max(0, 1.0 - dist / 20.0))

    return float(np.mean(scores)) if scores else 0.0


# ===========================================================================
# Tool 9: Batch Render
# ===========================================================================

@mcp.tool(
    name="batch_render",
    annotations={
        "title": "Batch Render with Multiple Parameter Sets",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    },
)
def batch_render(
    plugin_name: str,
    parameter_sets: str,
    input_path: Optional[str] = None,
    input_signal: Optional[str] = None,
    duration: float = 2.0,
    ctx: Context = None,
) -> str:
    """Render audio through a plugin with multiple parameter configurations.

    Useful for A/B testing, parameter space exploration, and dataset generation.

    Args:
        plugin_name: A loaded effect plugin.
        parameter_sets: JSON array of parameter dicts.
                        Example: '[{"ratio": 0.3}, {"ratio": 0.7}]'
        input_path: Path to input audio file. Shared across all renders.
        input_signal: Test signal JSON spec. Used if input_path is not provided.
        duration: Duration in seconds (for generated signals).

    Returns:
        List of output file paths with their parameter configs and stats.
    """
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return f"Plugin '{plugin_name}' not loaded."

    sr = state.sample_rate

    try:
        configs = json.loads(parameter_sets)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    if not isinstance(configs, list):
        return "parameter_sets must be a JSON array of objects."

    # Load or generate input audio
    if input_path:
        audio, sr = load_audio(input_path, sample_rate=sr)
    elif input_signal:
        spec = json.loads(input_signal)
        sig_type = spec.pop("type", "sine")
        audio = generate_test_signal(sig_type, duration=duration, sample_rate=sr,
                                      channels=state.channels, **spec)
    else:
        audio = generate_test_signal("sine", duration=duration, sample_rate=sr,
                                      channels=state.channels, freq=1000.0)

    results = []
    for i, config in enumerate(configs):
        try:
            inst.set_parameters_bulk(config, raw=True)
            output = inst.process_audio(audio, sample_rate=sr)
            out_path = _cache_path(f"{plugin_name}_batch_{i:03d}.wav")
            save_audio(output, out_path, sr)
            stats = audio_stats(output)
            results.append({
                "index": i,
                "parameters": config,
                "output_path": out_path,
                "peak_db": stats["peak_db"],
                "rms_db": stats["rms_db"],
            })
        except Exception as e:
            results.append({"index": i, "parameters": config, "error": str(e)})

    return json.dumps({"plugin": plugin_name, "renders": results}, indent=2)


# ===========================================================================
# Tool 10: Compile JUCE Project
# ===========================================================================

@mcp.tool(
    name="compile_juce_project",
    annotations={
        "title": "Compile JUCE Plugin Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    },
)
def compile_juce_project(
    project_path: str,
    build_type: str = "Release",
    ctx: Context = None,
) -> str:
    """Compile a JUCE audio plugin project from source using CMake.

    Requires CMake and a C++ compiler (MSVC on Windows, Xcode on macOS,
    GCC/Clang on Linux) to be installed on the system.

    Args:
        project_path: Path to the JUCE project directory containing CMakeLists.txt.
        build_type: 'Release' or 'Debug'.

    Returns:
        Build status, output paths for compiled plugins, and build log excerpt.
    """
    import platform
    import shutil
    import subprocess

    project_dir = Path(project_path).resolve()

    # Validate project
    if not (project_dir / "CMakeLists.txt").exists():
        return f"No CMakeLists.txt found in {project_dir}"

    # Check for CMake
    cmake = shutil.which("cmake")
    if not cmake:
        return "CMake not found. Install CMake and ensure it's in your PATH."

    build_dir = project_dir / "build"

    # Determine generator
    system = platform.system()
    configure_cmd = [cmake, "-S", str(project_dir), "-B", str(build_dir)]

    if system == "Windows":
        # Auto-detect Visual Studio
        configure_cmd.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])
    elif system == "Darwin":
        configure_cmd.extend(["-G", "Xcode"])
    # Linux: default generator (Unix Makefiles)

    configure_cmd.append(f"-DCMAKE_BUILD_TYPE={build_type}")

    # Configure
    try:
        config_result = subprocess.run(
            configure_cmd, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "CMake configure timed out after 120 seconds."
    except FileNotFoundError:
        return "CMake executable not found."

    if config_result.returncode != 0:
        return json.dumps({
            "status": "configure_failed",
            "returncode": config_result.returncode,
            "stderr": config_result.stderr[-2000:],
        }, indent=2)

    # Build
    build_cmd = [cmake, "--build", str(build_dir), "--config", build_type, "--parallel"]
    try:
        build_result = subprocess.run(
            build_cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "Build timed out after 600 seconds."

    if build_result.returncode != 0:
        return json.dumps({
            "status": "build_failed",
            "returncode": build_result.returncode,
            "stderr": build_result.stderr[-3000:],
        }, indent=2)

    # Find built plugin files
    artefacts_dir = build_dir / "*_artefacts" 
    plugins_found = []
    for ext in ["*.vst3", "*.component", "*.clap", "*.lv2"]:
        plugins_found.extend(str(p) for p in build_dir.rglob(ext))

    return json.dumps({
        "status": "success",
        "project": str(project_dir),
        "build_type": build_type,
        "plugins_found": plugins_found,
        "build_log_tail": build_result.stdout[-1000:],
    }, indent=2)


# ===========================================================================
# Tool 11: Scan Plugins
# ===========================================================================

@mcp.tool(
    name="scan_plugins",
    annotations={
        "title": "Scan Plugin Directories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
def scan_plugins(
    directory: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Scan a directory (or all configured directories) for audio plugins.

    Args:
        directory: Specific directory to scan. If not provided, scans all
                   directories listed in config.yaml plugin_search_paths.

    Returns:
        List of found plugins with name, path, format, and size.
    """
    state = _get_state(ctx)

    if directory:
        dirs = [directory]
    else:
        dirs = CONFIG.get("plugin_search_paths", [])

    all_plugins = []
    for d in dirs:
        expanded = os.path.expanduser(d)
        if os.path.isdir(expanded):
            found = state.engine.scan_directory(expanded)
            all_plugins.extend(found)

    return json.dumps({
        "directories_scanned": len(dirs),
        "plugins_found": len(all_plugins),
        "plugins": all_plugins,
    }, indent=2)


# ===========================================================================
# NEW TOOLS: Audio File Operations
# ===========================================================================

@mcp.tool(
    name="convert_audio",
    description="Convert audio between formats, sample rates, bit depths, and channel configurations. "
                "Supports WAV, FLAC, OGG, MP3 (input only). Output is always WAV or FLAC.",
)
def convert_audio(
    ctx: Context,
    input_path: str,
    output_format: str = "wav",
    sample_rate: int | None = None,
    bit_depth: int = 32,
    mono: bool = False,
    stereo: bool = False,
    normalize: bool = False,
) -> str:
    """Convert audio file format, sample rate, bit depth, or channel count."""
    audio, sr = load_audio(input_path, sample_rate=sample_rate, mono=mono)

    if stereo and audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak * 0.99

    out_sr = sample_rate or sr
    ext = "flac" if output_format.lower() == "flac" else "wav"
    out_path = _cache_path(f"converted_{uuid.uuid4().hex[:8]}.{ext}")
    save_audio(audio, out_path, out_sr, bit_depth=bit_depth)

    stats = audio_stats(audio)
    return json.dumps({
        "output_path": out_path,
        "format": ext.upper(),
        "sample_rate": out_sr,
        "bit_depth": bit_depth,
        "channels": audio.shape[0],
        "samples": audio.shape[1],
        "duration_seconds": round(audio.shape[1] / out_sr, 3),
        "peak_db": stats["peak_db"],
    }, indent=2)


@mcp.tool(
    name="trim_audio",
    description="Trim, fade, normalize, or split audio files. Specify start/end times in seconds, "
                "fade in/out durations, normalization target, or split-at-silence threshold.",
)
def trim_audio(
    ctx: Context,
    input_path: str,
    start: float = 0.0,
    end: float | None = None,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    normalize_db: float | None = None,
    split_silence_threshold_db: float | None = None,
    split_silence_min_duration: float = 0.5,
) -> str:
    """Trim, fade, normalize, or split audio at silence boundaries."""
    audio, sr = load_audio(input_path)

    start_sample = int(start * sr)
    end_sample = int(end * sr) if end is not None else audio.shape[1]
    end_sample = min(end_sample, audio.shape[1])
    audio = audio[:, start_sample:end_sample]

    # Fade in
    if fade_in > 0:
        fade_samples = min(int(fade_in * sr), audio.shape[1])
        fade_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        audio[:, :fade_samples] *= fade_curve

    # Fade out
    if fade_out > 0:
        fade_samples = min(int(fade_out * sr), audio.shape[1])
        fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        audio[:, -fade_samples:] *= fade_curve

    # Normalize
    if normalize_db is not None:
        peak = np.max(np.abs(audio))
        if peak > 1e-10:
            target_linear = 10 ** (normalize_db / 20.0)
            audio = audio * (target_linear / peak)
            audio = np.clip(audio, -1.0, 1.0)

    # Split at silence
    if split_silence_threshold_db is not None:
        threshold = 10 ** (split_silence_threshold_db / 20.0)
        min_samples = int(split_silence_min_duration * sr)
        mono_signal = np.mean(np.abs(audio), axis=0)
        is_silent = mono_signal < threshold

        segments = []
        segment_start = 0
        silent_count = 0
        in_silence = False

        for i in range(len(is_silent)):
            if is_silent[i]:
                silent_count += 1
                if silent_count >= min_samples and not in_silence:
                    if i - silent_count > segment_start:
                        segments.append((segment_start, i - silent_count))
                    in_silence = True
            else:
                if in_silence:
                    segment_start = i
                    in_silence = False
                silent_count = 0

        if not in_silence and segment_start < audio.shape[1]:
            segments.append((segment_start, audio.shape[1]))

        output_paths = []
        for idx, (seg_start, seg_end) in enumerate(segments):
            seg_path = _cache_path(f"segment_{idx:03d}_{uuid.uuid4().hex[:6]}.wav")
            save_audio(audio[:, seg_start:seg_end], seg_path, sr)
            output_paths.append(seg_path)

        return json.dumps({
            "operation": "split_at_silence",
            "segments_found": len(segments),
            "output_paths": output_paths,
            "threshold_db": split_silence_threshold_db,
        }, indent=2)

    # Single output
    out_path = _cache_path(f"trimmed_{uuid.uuid4().hex[:8]}.wav")
    save_audio(audio, out_path, sr)
    stats = audio_stats(audio)

    return json.dumps({
        "output_path": out_path,
        "duration_seconds": round(audio.shape[1] / sr, 3),
        "peak_db": stats["peak_db"],
        "rms_db": stats["rms_db"],
    }, indent=2)


@mcp.tool(
    name="concatenate_audio",
    description="Join multiple audio files sequentially (concat) or layer them together (mix). "
                "Optionally add crossfade between concatenated segments.",
)
def concatenate_audio(
    ctx: Context,
    input_paths: list[str],
    mode: str = "concat",
    crossfade: float = 0.0,
    normalize_output: bool = False,
) -> str:
    """Join audio files end-to-end (concat) or layer them (mix)."""
    if not input_paths:
        return json.dumps({"error": "No input paths provided"})

    segments = []
    target_sr = None

    for path in input_paths:
        audio, sr = load_audio(path)
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            import librosa
            resampled = []
            for ch in range(audio.shape[0]):
                resampled.append(librosa.resample(audio[ch], orig_sr=sr, target_sr=target_sr))
            audio = np.stack(resampled)
        segments.append(audio)

    # Match channel counts
    max_ch = max(s.shape[0] for s in segments)
    segments = [np.repeat(s, max_ch // s.shape[0] + 1, axis=0)[:max_ch] if s.shape[0] < max_ch else s for s in segments]

    if mode == "mix":
        max_len = max(s.shape[1] for s in segments)
        mixed = np.zeros((max_ch, max_len), dtype=np.float32)
        for seg in segments:
            mixed[:, :seg.shape[1]] += seg
        # Prevent clipping
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed /= peak
        result = mixed
    else:
        # Concatenate with optional crossfade
        if crossfade > 0 and len(segments) > 1:
            xfade_samples = int(crossfade * target_sr)
            parts = [segments[0]]
            for seg in segments[1:]:
                prev = parts[-1]
                if xfade_samples > 0 and xfade_samples < min(prev.shape[1], seg.shape[1]):
                    fade_out = np.linspace(1.0, 0.0, xfade_samples, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)
                    overlap = prev[:, -xfade_samples:] * fade_out + seg[:, :xfade_samples] * fade_in
                    parts[-1] = prev[:, :-xfade_samples]
                    parts.append(overlap)
                    parts.append(seg[:, xfade_samples:])
                else:
                    parts.append(seg)
            result = np.concatenate(parts, axis=1)
        else:
            result = np.concatenate(segments, axis=1)

    if normalize_output:
        peak = np.max(np.abs(result))
        if peak > 1e-6:
            result = result / peak * 0.99

    out_path = _cache_path(f"{mode}_{uuid.uuid4().hex[:8]}.wav")
    save_audio(result, out_path, target_sr)

    return json.dumps({
        "output_path": out_path,
        "mode": mode,
        "input_count": len(input_paths),
        "duration_seconds": round(result.shape[1] / target_sr, 3),
        "channels": result.shape[0],
        "peak_db": round(20 * np.log10(max(float(np.max(np.abs(result))), 1e-10)), 2),
    }, indent=2)


# ===========================================================================
# NEW TOOLS: Plugin Chain Management
# ===========================================================================

@mcp.tool(
    name="create_chain",
    description="Build a multi-plugin serial processing chain and render audio through it in one call. "
                "Specify an ordered list of loaded plugin names. Audio passes through each in sequence.",
)
def create_chain(
    ctx: Context,
    plugin_names: list[str],
    input_path: str | None = None,
    input_signal: str | None = None,
    signal_duration: float = 5.0,
) -> str:
    """Process audio through a chain of loaded plugins in series."""
    state = _get_state(ctx)

    # Get input audio
    if input_path:
        audio, sr = load_audio(input_path, sample_rate=state.sample_rate)
    elif input_signal:
        audio = generate_test_signal(input_signal, duration=signal_duration,
                                     sample_rate=state.sample_rate, channels=CHANNELS)
        sr = state.sample_rate
    else:
        return json.dumps({"error": "Provide input_path or input_signal"})

    chain_log = []
    for name in plugin_names:
        inst = state.engine.get(name)
        if inst is None:
            return json.dumps({"error": f"Plugin '{name}' is not loaded"})

        pre_stats = audio_stats(audio)
        audio = inst.process_audio(audio, sr)
        post_stats = audio_stats(audio)

        chain_log.append({
            "plugin": name,
            "input_peak_db": pre_stats["peak_db"],
            "output_peak_db": post_stats["peak_db"],
        })

    out_path = _cache_path(f"chain_{uuid.uuid4().hex[:8]}.wav")
    save_audio(audio, out_path, sr)

    return json.dumps({
        "output_path": out_path,
        "chain": chain_log,
        "total_plugins": len(plugin_names),
        "duration_seconds": round(audio.shape[1] / sr, 3),
        "final_peak_db": audio_stats(audio)["peak_db"],
    }, indent=2)


@mcp.tool(
    name="save_plugin_state",
    description="Export a loaded plugin's complete state (all parameter values) to a JSON file. "
                "Use load_plugin_state to restore it later.",
)
def save_plugin_state(
    ctx: Context,
    plugin_name: str,
    output_path: str | None = None,
    preset_name: str = "default",
) -> str:
    """Save all parameter values of a loaded plugin to a JSON file."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    params = inst.get_parameters()
    preset_data = {
        "plugin_name": inst.name,
        "plugin_path": inst.path,
        "preset_name": preset_name,
        "parameters": {p["name"]: p["raw_value"] for p in params},
    }

    if output_path is None:
        output_path = _cache_path(f"state_{plugin_name}_{preset_name}_{uuid.uuid4().hex[:6]}.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(preset_data, f, indent=2)

    return json.dumps({
        "output_path": output_path,
        "plugin": plugin_name,
        "preset_name": preset_name,
        "parameter_count": len(params),
    }, indent=2)


@mcp.tool(
    name="load_plugin_state",
    description="Restore a plugin's parameter values from a previously saved state JSON file.",
)
def load_plugin_state(
    ctx: Context,
    plugin_name: str,
    state_path: str,
) -> str:
    """Load parameter values from a JSON state file into a loaded plugin."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    with open(state_path, "r") as f:
        preset_data = json.load(f)

    params = preset_data.get("parameters", {})
    results = inst.set_parameters_bulk(params, raw=True)

    return json.dumps({
        "plugin": plugin_name,
        "preset_loaded": preset_data.get("preset_name", "unknown"),
        "parameters_set": len(results),
        "details": results,
    }, indent=2)


@mcp.tool(
    name="get_plugin_latency",
    description="Report the latency of a loaded plugin in samples and milliseconds.",
)
def get_plugin_latency(
    ctx: Context,
    plugin_name: str,
    sample_rate: int | None = None,
) -> str:
    """Get the reported latency of a loaded plugin."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    sr = sample_rate or state.sample_rate
    latency_samples = getattr(inst.plugin, "latency_samples", 0) or 0
    latency_ms = round(latency_samples / sr * 1000, 3) if latency_samples else 0.0

    return json.dumps({
        "plugin": plugin_name,
        "latency_samples": latency_samples,
        "latency_ms": latency_ms,
        "sample_rate": sr,
    }, indent=2)


# ===========================================================================
# NEW TOOLS: Advanced Analysis
# ===========================================================================

@mcp.tool(
    name="extract_impulse_response",
    description="Send an impulse through a loaded effect plugin and capture the impulse response as a WAV. "
                "Useful for analyzing reverbs, delays, and any linear time-invariant processing.",
)
def extract_impulse_response(
    ctx: Context,
    plugin_name: str,
    duration: float = 3.0,
) -> str:
    """Extract impulse response from an effect plugin."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    sr = state.sample_rate
    impulse = generate_test_signal("impulse", duration=duration, sample_rate=sr, channels=CHANNELS)
    ir = inst.process_audio(impulse, sr)

    out_path = _cache_path(f"ir_{plugin_name}_{uuid.uuid4().hex[:6]}.wav")
    save_audio(ir, out_path, sr)

    stats = audio_stats(ir)
    # Estimate RT60 (time for signal to decay by 60dB)
    mono_ir = np.mean(np.abs(ir), axis=0)
    peak_idx = np.argmax(mono_ir)
    peak_val = mono_ir[peak_idx]
    threshold = peak_val * 0.001  # -60dB
    decay_indices = np.where(mono_ir[peak_idx:] < threshold)[0]
    rt60 = round(decay_indices[0] / sr, 3) if len(decay_indices) > 0 else duration

    return json.dumps({
        "output_path": out_path,
        "plugin": plugin_name,
        "duration_seconds": duration,
        "peak_db": stats["peak_db"],
        "estimated_rt60": rt60,
        "sample_rate": sr,
    }, indent=2)


@mcp.tool(
    name="measure_thd",
    description="Measure Total Harmonic Distortion of a plugin by feeding a pure sine wave "
                "and analyzing the harmonic content of the output. Returns THD percentage and per-harmonic levels.",
)
def measure_thd(
    ctx: Context,
    plugin_name: str,
    frequency: float = 1000.0,
    amplitude: float = 0.5,
    duration: float = 2.0,
    num_harmonics: int = 10,
) -> str:
    """Measure THD by sending a sine and analyzing harmonics."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    sr = state.sample_rate
    sine = generate_test_signal("sine", duration=duration, sample_rate=sr,
                                channels=CHANNELS, freq=frequency, amplitude=amplitude)
    output = inst.process_audio(sine, sr)

    # Analyze first channel
    signal = output[0]
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * window)) / (n / 2)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Find fundamental and harmonics
    harmonics = {}
    fundamental_power = 0.0
    harmonic_power = 0.0

    for h in range(1, num_harmonics + 1):
        target_freq = frequency * h
        if target_freq >= sr / 2:
            break
        idx = np.argmin(np.abs(freqs - target_freq))
        # Sum energy in a small window around the harmonic
        window_size = max(1, int(10 * n / sr))
        start = max(0, idx - window_size)
        end = min(len(spectrum), idx + window_size + 1)
        level = float(np.max(spectrum[start:end]))
        level_db = round(20 * np.log10(max(level, 1e-10)), 2)

        harmonics[f"H{h} ({target_freq:.0f}Hz)"] = level_db

        if h == 1:
            fundamental_power = level ** 2
        else:
            harmonic_power += level ** 2

    thd_percent = round(np.sqrt(harmonic_power / max(fundamental_power, 1e-20)) * 100, 4) if fundamental_power > 0 else 0.0

    return json.dumps({
        "plugin": plugin_name,
        "test_frequency": frequency,
        "input_amplitude": amplitude,
        "thd_percent": thd_percent,
        "thd_db": round(20 * np.log10(max(thd_percent / 100, 1e-10)), 2),
        "harmonics": harmonics,
    }, indent=2)


@mcp.tool(
    name="detect_aliasing",
    description="Feed high-frequency content through a plugin and check for spectral energy "
                "below the expected range, indicating aliased frequencies folding back below Nyquist.",
)
def detect_aliasing(
    ctx: Context,
    plugin_name: str,
    test_frequency: float | None = None,
    duration: float = 2.0,
) -> str:
    """Detect aliasing by analyzing spectral content below input frequency range."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    sr = state.sample_rate
    nyquist = sr / 2.0
    freq = test_frequency or (nyquist * 0.4)  # Default: 40% of Nyquist

    sine = generate_test_signal("sine", duration=duration, sample_rate=sr,
                                channels=CHANNELS, freq=freq, amplitude=0.5)
    output = inst.process_audio(sine, sr)

    signal = output[0]
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n))) / (n / 2)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Measure energy in expected band (around fundamental and harmonics that fit)
    expected_energy = 0.0
    alias_energy = 0.0
    noise_floor_energy = 0.0

    for i, f in enumerate(freqs):
        power = spectrum[i] ** 2
        is_expected = any(abs(f - freq * h) < 50 for h in range(1, 20) if freq * h < nyquist)
        if is_expected:
            expected_energy += power
        elif f > 100 and f < freq * 0.8:
            alias_energy += power
        elif f > 100:
            noise_floor_energy += power

    alias_ratio = alias_energy / max(expected_energy, 1e-20)
    alias_db = round(10 * np.log10(max(alias_ratio, 1e-20)), 2)

    return json.dumps({
        "plugin": plugin_name,
        "test_frequency": freq,
        "sample_rate": sr,
        "nyquist": nyquist,
        "alias_to_signal_ratio_db": alias_db,
        "aliasing_detected": alias_db > -60,
        "severity": "none" if alias_db < -80 else "minimal" if alias_db < -60 else "moderate" if alias_db < -40 else "severe",
        "recommendation": "Clean" if alias_db < -60 else "Consider 2x oversampling" if alias_db < -40 else "Oversampling strongly recommended",
    }, indent=2)


@mcp.tool(
    name="measure_frequency_response",
    description="Sweep a sine wave across the frequency spectrum through a plugin and measure "
                "the output level at each frequency. Returns the magnitude response curve.",
)
def measure_frequency_response(
    ctx: Context,
    plugin_name: str,
    start_freq: float = 20.0,
    end_freq: float = 20000.0,
    num_points: int = 64,
    amplitude: float = 0.5,
) -> str:
    """Measure frequency response by sweeping sine tones."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    sr = state.sample_rate
    end_freq = min(end_freq, sr / 2 - 100)

    # Generate log-spaced frequency points
    test_freqs = np.geomspace(start_freq, end_freq, num_points)
    response = []

    for freq in test_freqs:
        sine = generate_test_signal("sine", duration=0.5, sample_rate=sr,
                                    channels=CHANNELS, freq=float(freq), amplitude=amplitude)
        output = inst.process_audio(sine, sr)

        # Measure output RMS (skip first 1000 samples for transients)
        signal = output[0, 1000:] if output.shape[1] > 2000 else output[0]
        rms = float(np.sqrt(np.mean(signal ** 2)))
        level_db = round(20 * np.log10(max(rms, 1e-10)), 2)
        input_rms = float(np.sqrt(np.mean(sine[0, 1000:] ** 2))) if sine.shape[1] > 2000 else amplitude / np.sqrt(2)
        gain_db = round(level_db - 20 * np.log10(max(input_rms, 1e-10)), 2)

        response.append({
            "frequency": round(float(freq), 1),
            "output_db": level_db,
            "gain_db": gain_db,
        })

    return json.dumps({
        "plugin": plugin_name,
        "measurement_points": len(response),
        "frequency_range": [start_freq, end_freq],
        "input_amplitude": amplitude,
        "response": response,
    }, indent=2)


# ===========================================================================
# NEW TOOLS: MIDI Utilities
# ===========================================================================

@mcp.tool(
    name="generate_midi_file",
    description="Create a MIDI file from a text description of notes. Supports note names (C4, D#5), "
                "chords, scales, and timing in beats at a given BPM.",
)
def generate_midi_file(
    ctx: Context,
    notes: list[dict],
    bpm: float = 120.0,
    output_path: str | None = None,
) -> str:
    """Create a MIDI file. Each note dict: {note, velocity, start_beat, duration_beats}."""
    import mido

    midi_file = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))

    # Build events
    events = []
    for n in notes:
        note_name = n.get("note", "C4")
        velocity = n.get("velocity", 100)
        start_beat = n.get("start_beat", 0.0)
        dur_beats = n.get("duration_beats", 1.0)

        # Parse note name to MIDI number
        if isinstance(note_name, int):
            midi_num = note_name
        else:
            midi_num = _note_name_to_midi(note_name)

        start_tick = int(start_beat * 480)
        end_tick = int((start_beat + dur_beats) * 480)

        events.append((start_tick, "note_on", midi_num, velocity))
        events.append((end_tick, "note_off", midi_num, 0))

    events.sort(key=lambda x: x[0])

    # Convert to delta times
    prev_tick = 0
    for tick, msg_type, note, vel in events:
        delta = tick - prev_tick
        track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
        prev_tick = tick

    if output_path is None:
        output_path = _cache_path(f"midi_{uuid.uuid4().hex[:8]}.mid")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(output_path)

    return json.dumps({
        "output_path": output_path,
        "bpm": bpm,
        "note_count": len(notes),
        "duration_beats": max(n.get("start_beat", 0) + n.get("duration_beats", 1) for n in notes) if notes else 0,
    }, indent=2)


def _note_name_to_midi(name: str) -> int:
    """Convert note name like 'C4', 'D#5', 'Bb3' to MIDI number."""
    note_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    name = name.strip()
    idx = 0
    base = note_map.get(name[idx].upper(), 0)
    idx += 1
    if idx < len(name) and name[idx] in "#♯":
        base += 1
        idx += 1
    elif idx < len(name) and name[idx] in "b♭":
        base -= 1
        idx += 1
    octave = int(name[idx:]) if idx < len(name) else 4
    return base + (octave + 1) * 12


@mcp.tool(
    name="analyze_midi",
    description="Parse a MIDI file and report note count, pitch range, velocity distribution, "
                "timing info, channels used, and duration.",
)
def analyze_midi(
    ctx: Context,
    midi_path: str,
) -> str:
    """Analyze a MIDI file and return statistics."""
    import mido

    midi_file = mido.MidiFile(midi_path)

    notes = []
    tempos = []
    channels_used = set()

    for track in midi_file.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append({"note": msg.note, "velocity": msg.velocity,
                              "time_ticks": abs_time, "channel": msg.channel})
                channels_used.add(msg.channel)
            elif msg.type == "set_tempo":
                tempos.append(mido.tempo2bpm(msg.tempo))

    velocities = [n["velocity"] for n in notes]
    pitches = [n["note"] for n in notes]

    return json.dumps({
        "file": midi_path,
        "tracks": len(midi_file.tracks),
        "ticks_per_beat": midi_file.ticks_per_beat,
        "duration_seconds": round(midi_file.length, 2),
        "note_count": len(notes),
        "channels_used": sorted(channels_used),
        "pitch_range": {"lowest": min(pitches), "highest": max(pitches)} if pitches else None,
        "velocity_stats": {
            "min": min(velocities),
            "max": max(velocities),
            "mean": round(sum(velocities) / len(velocities), 1),
        } if velocities else None,
        "tempos_bpm": [round(t, 1) for t in tempos] if tempos else [120.0],
    }, indent=2)


# ===========================================================================
# NEW TOOLS: Preset Management
# ===========================================================================

@mcp.tool(
    name="export_preset_bank",
    description="Export multiple parameter configurations as a preset bank JSON file. "
                "Each preset is a named set of parameter values.",
)
def export_preset_bank(
    ctx: Context,
    plugin_name: str,
    presets: list[dict],
    output_path: str | None = None,
) -> str:
    """Export a bank of presets. Each dict: {name, parameters: {param: value}}."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    bank = {
        "plugin_name": inst.name,
        "plugin_path": inst.path,
        "preset_count": len(presets),
        "presets": presets,
    }

    if output_path is None:
        output_path = _cache_path(f"bank_{plugin_name}_{uuid.uuid4().hex[:6]}.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bank, f, indent=2)

    return json.dumps({
        "output_path": output_path,
        "plugin": plugin_name,
        "preset_count": len(presets),
        "preset_names": [p.get("name", f"preset_{i}") for i, p in enumerate(presets)],
    }, indent=2)


@mcp.tool(
    name="randomize_parameters",
    description="Generate random but musically sensible parameter values for a loaded plugin. "
                "Respects parameter ranges. Optionally constrain to a 'neighborhood' around current values.",
)
def randomize_parameters(
    ctx: Context,
    plugin_name: str,
    variance: float = 1.0,
    apply: bool = False,
    seed: int | None = None,
) -> str:
    """Generate random parameter values. Variance 0-1 controls how far from current values."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    rng = np.random.default_rng(seed)
    params = inst.get_parameters()
    randomized = {}

    for p in params:
        name = p["name"]
        current = p["raw_value"]

        if p.get("is_boolean", False):
            new_val = float(rng.choice([0.0, 1.0]))
        elif variance >= 1.0:
            new_val = float(rng.uniform(0.0, 1.0))
        else:
            offset = float(rng.normal(0, variance * 0.3))
            new_val = max(0.0, min(1.0, current + offset))

        randomized[name] = round(new_val, 4)

    if apply:
        inst.set_parameters_bulk(randomized, raw=True)

    return json.dumps({
        "plugin": plugin_name,
        "variance": variance,
        "applied": apply,
        "parameters": randomized,
    }, indent=2)


@mcp.tool(
    name="interpolate_presets",
    description="Morph between two parameter configurations by a blend factor (0.0 = preset A, "
                "1.0 = preset B, 0.5 = halfway). Optionally apply the result to the loaded plugin.",
)
def interpolate_presets(
    ctx: Context,
    plugin_name: str,
    preset_a: dict[str, float],
    preset_b: dict[str, float],
    blend: float = 0.5,
    apply: bool = False,
) -> str:
    """Interpolate between two parameter sets."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    blend = max(0.0, min(1.0, blend))
    all_keys = set(list(preset_a.keys()) + list(preset_b.keys()))
    interpolated = {}

    for key in all_keys:
        val_a = preset_a.get(key, 0.5)
        val_b = preset_b.get(key, 0.5)
        interpolated[key] = round(val_a + (val_b - val_a) * blend, 6)

    if apply:
        inst.set_parameters_bulk(interpolated, raw=True)

    return json.dumps({
        "plugin": plugin_name,
        "blend": blend,
        "applied": apply,
        "parameters": interpolated,
    }, indent=2)


# ===========================================================================
# NEW TOOLS: Monitoring & Testing
# ===========================================================================

@mcp.tool(
    name="monitor_cpu_usage",
    description="Measure how much CPU time a plugin consumes during processing at various buffer sizes. "
                "Reports real-time ratio (< 1.0 means real-time safe) and absolute processing time.",
)
def monitor_cpu_usage(
    ctx: Context,
    plugin_name: str,
    buffer_sizes: list[int] | None = None,
    iterations: int = 100,
    signal_type: str = "white_noise",
) -> str:
    """Benchmark plugin CPU usage across buffer sizes."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    if buffer_sizes is None:
        buffer_sizes = [64, 128, 256, 512, 1024]

    sr = state.sample_rate
    results = []

    for buf_size in buffer_sizes:
        # Generate test audio for this buffer size
        duration = buf_size / sr
        audio = generate_test_signal(signal_type, duration=duration,
                                     sample_rate=sr, channels=CHANNELS)

        # Warm up
        for _ in range(5):
            inst.process_audio(audio, sr)

        # Measure
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            inst.process_audio(audio, sr)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        budget = buf_size / sr
        avg_time = sum(times) / len(times)
        max_time = max(times)
        p99_time = sorted(times)[int(len(times) * 0.99)]

        results.append({
            "buffer_size": buf_size,
            "budget_ms": round(budget * 1000, 3),
            "avg_ms": round(avg_time * 1000, 3),
            "max_ms": round(max_time * 1000, 3),
            "p99_ms": round(p99_time * 1000, 3),
            "avg_realtime_ratio": round(avg_time / budget, 4),
            "max_realtime_ratio": round(max_time / budget, 4),
            "realtime_safe": max_time < budget,
        })

    return json.dumps({
        "plugin": plugin_name,
        "sample_rate": sr,
        "iterations_per_size": iterations,
        "signal_type": signal_type,
        "results": results,
    }, indent=2)


@mcp.tool(
    name="stress_test_plugin",
    description="Hammer a plugin with edge cases: silence, max level, DC offset, very small buffers, "
                "very large buffers, rapid parameter changes, and extreme parameter values. "
                "Reports any crashes, NaN/Inf output, or anomalies.",
)
def stress_test_plugin(
    ctx: Context,
    plugin_name: str,
) -> str:
    """Run a comprehensive stress test battery on a loaded plugin."""
    state = _get_state(ctx)
    inst = state.engine.get(plugin_name)
    if inst is None:
        return json.dumps({"error": f"Plugin '{plugin_name}' not loaded"})

    sr = state.sample_rate
    test_results = []

    def run_test(name: str, audio: np.ndarray) -> dict:
        result = {"test": name, "passed": True, "issue": None}
        try:
            output = inst.process_audio(audio, sr)
            if np.any(np.isnan(output)):
                result["passed"] = False
                result["issue"] = "Output contains NaN"
            elif np.any(np.isinf(output)):
                result["passed"] = False
                result["issue"] = "Output contains Inf"
            elif np.max(np.abs(output)) > 10.0:
                result["passed"] = False
                result["issue"] = f"Output level extreme: {float(np.max(np.abs(output))):.1f}"
        except Exception as e:
            result["passed"] = False
            result["issue"] = f"Exception: {str(e)[:200]}"
        return result

    # Test 1: Silence
    test_results.append(run_test("silence", np.zeros((CHANNELS, sr), dtype=np.float32)))

    # Test 2: Max level (+0 dBFS)
    test_results.append(run_test("max_level", np.ones((CHANNELS, sr), dtype=np.float32)))

    # Test 3: DC offset
    dc = np.full((CHANNELS, sr), 0.5, dtype=np.float32)
    test_results.append(run_test("dc_offset", dc))

    # Test 4: Very short buffer (32 samples)
    test_results.append(run_test("tiny_buffer_32", np.random.randn(CHANNELS, 32).astype(np.float32) * 0.5))

    # Test 5: Single sample
    test_results.append(run_test("single_sample", np.random.randn(CHANNELS, 1).astype(np.float32) * 0.5))

    # Test 6: Large buffer (65536 samples)
    test_results.append(run_test("large_buffer_65536", np.random.randn(CHANNELS, 65536).astype(np.float32) * 0.5))

    # Test 7: Extreme values (clipped at ±10)
    extreme = np.random.randn(CHANNELS, sr).astype(np.float32) * 10.0
    test_results.append(run_test("extreme_values", extreme))

    # Test 8: Denormal territory
    denormal = np.full((CHANNELS, sr), 1e-38, dtype=np.float32)
    test_results.append(run_test("denormal_values", denormal))

    # Test 9: Rapid parameter changes
    params = inst.get_parameters()
    rapid_result = {"test": "rapid_param_changes", "passed": True, "issue": None}
    try:
        audio = np.random.randn(CHANNELS, sr).astype(np.float32) * 0.3
        for _ in range(50):
            for p in params[:5]:  # First 5 params
                try:
                    inst.set_parameter(p["name"], float(np.random.uniform(0, 1)), raw=True)
                except Exception:
                    pass
            inst.process_audio(audio[:, :128], sr)
        rapid_result["passed"] = True
    except Exception as e:
        rapid_result["passed"] = False
        rapid_result["issue"] = f"Exception: {str(e)[:200]}"
    test_results.append(rapid_result)

    # Test 10: All parameters at minimum (0.0)
    try:
        for p in params:
            try:
                inst.set_parameter(p["name"], 0.0, raw=True)
            except Exception:
                pass
    except Exception:
        pass
    audio = np.random.randn(CHANNELS, sr // 2).astype(np.float32) * 0.5
    test_results.append(run_test("all_params_minimum", audio))

    # Test 11: All parameters at maximum (1.0)
    try:
        for p in params:
            try:
                inst.set_parameter(p["name"], 1.0, raw=True)
            except Exception:
                pass
    except Exception:
        pass
    test_results.append(run_test("all_params_maximum", audio))

    passed = sum(1 for t in test_results if t["passed"])
    total = len(test_results)

    return json.dumps({
        "plugin": plugin_name,
        "tests_run": total,
        "tests_passed": passed,
        "tests_failed": total - passed,
        "all_passed": passed == total,
        "results": test_results,
    }, indent=2)


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        # FastMCP reads host/port from these env vars
        os.environ.setdefault("MCP_HOST", "0.0.0.0")
        os.environ.setdefault("MCP_PORT", os.environ.get("PORT", "8000"))
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")

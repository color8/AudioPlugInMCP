"""Pedalboard engine — VST3/AU plugin hosting and audio processing.

Wraps Spotify's pedalboard library for loading external plugins, inspecting
parameters, setting values, and processing audio. This is the primary engine
for effect processing and basic instrument rendering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pedalboard import Pedalboard, load_plugin
from pedalboard.io import AudioFile


class PluginInstance:
    """Wraps a loaded pedalboard plugin with metadata and convenience methods."""

    def __init__(self, plugin: Any, name: str, path: str):
        self.plugin = plugin
        self.name = name
        self.path = path

    @property
    def is_instrument(self) -> bool:
        return self.plugin.is_instrument

    @property
    def is_effect(self) -> bool:
        return self.plugin.is_effect

    def get_parameters(self) -> list[dict]:
        """Return all parameters with their current values and metadata."""
        params = []
        for param_name, param in self.plugin.parameters.items():
            info = {
                "name": param_name,
                "raw_value": round(float(param.raw_value), 6),
                "label": getattr(param, "label", ""),
                "is_discrete": getattr(param, "is_discrete", False),
                "is_boolean": getattr(param, "is_boolean", False),
                "is_automatable": getattr(param, "is_automatable", True),
            }

            # Try to get the real-world value via attribute access
            try:
                real_value = getattr(self.plugin, param_name.replace(" ", "_"), None)
                if real_value is not None:
                    info["real_value"] = round(float(real_value), 6)
            except (TypeError, ValueError):
                pass

            # Get string representation if available
            try:
                info["display_value"] = str(param)
            except Exception:
                pass

            params.append(info)

        return params

    def set_parameter(self, name: str, value: float, raw: bool = False) -> dict:
        """Set a parameter by name.

        Args:
            name: Parameter name (as returned by get_parameters).
            value: The value to set. If raw=True, expects 0.0-1.0 normalized.
            raw: If True, set via raw_value. If False, set via attribute.

        Returns:
            Dict with the parameter name and its new value after setting.
        """
        if raw:
            if name in self.plugin.parameters:
                self.plugin.parameters[name].raw_value = value
            else:
                raise ValueError(f"Parameter '{name}' not found")
        else:
            # Try attribute-style access first (uses real-world values)
            attr_name = name.replace(" ", "_")
            if hasattr(self.plugin, attr_name):
                setattr(self.plugin, attr_name, value)
            elif name in self.plugin.parameters:
                self.plugin.parameters[name].raw_value = value
            else:
                raise ValueError(f"Parameter '{name}' not found")

        # Read back actual value
        actual = self.plugin.parameters[name].raw_value if name in self.plugin.parameters else value
        return {"name": name, "set_value": value, "actual_raw_value": round(float(actual), 6)}

    def set_parameters_bulk(self, params: dict[str, float], raw: bool = False) -> list[dict]:
        """Set multiple parameters at once."""
        return [self.set_parameter(name, val, raw=raw) for name, val in params.items()]

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        buffer_size: int = 8192,
    ) -> np.ndarray:
        """Process audio through this effect plugin.

        Args:
            audio: Input audio shaped (channels, samples) as float32.
            sample_rate: Sample rate in Hz.
            buffer_size: Processing buffer size.

        Returns:
            Processed audio as float32 (channels, samples).
        """
        if self.is_instrument:
            raise RuntimeError(
                f"'{self.name}' is an instrument plugin. Use render_midi() instead."
            )

        return self.plugin.process(
            input_array=audio,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            reset=True,
        )

    def render_midi(
        self,
        midi_messages: list,
        duration: float = 5.0,
        sample_rate: int = 44100,
        num_channels: int = 2,
        buffer_size: int = 8192,
    ) -> np.ndarray:
        """Render MIDI through this instrument plugin.

        Args:
            midi_messages: List of mido.Message objects with absolute `time` in seconds.
            duration: Total output duration in seconds.
            sample_rate: Sample rate in Hz.
            num_channels: Number of output channels.
            buffer_size: Processing buffer size.

        Returns:
            Rendered audio as float32 (channels, samples).
        """
        if not self.is_instrument:
            raise RuntimeError(
                f"'{self.name}' is an effect plugin. Use process_audio() instead."
            )

        return self.plugin(
            midi_messages,
            duration=duration,
            sample_rate=sample_rate,
            num_channels=num_channels,
            buffer_size=buffer_size,
        )


class PedalboardEngine:
    """Manages plugin loading, caching, and audio processing via Pedalboard."""

    def __init__(self):
        self._plugins: dict[str, PluginInstance] = {}

    @property
    def loaded_plugins(self) -> dict[str, PluginInstance]:
        return self._plugins

    def load(
        self,
        path: str,
        name: str | None = None,
        parameter_values: dict | None = None,
        plugin_name: str | None = None,
    ) -> PluginInstance:
        """Load a VST3/AU plugin from disk.

        Args:
            path: File path to the .vst3 or .component plugin.
            name: Friendly name for referencing later. Defaults to filename stem.
            parameter_values: Optional initial parameter values.
            plugin_name: For multi-plugin bundles, specify which plugin to load.

        Returns:
            A PluginInstance wrapping the loaded plugin.
        """
        from pathlib import Path as P

        if name is None:
            name = P(path).stem

        # Avoid reloading if already loaded with same path
        if name in self._plugins and self._plugins[name].path == path:
            return self._plugins[name]

        plugin = load_plugin(
            path_to_plugin_file=path,
            parameter_values=parameter_values or {},
            plugin_name=plugin_name,
        )

        instance = PluginInstance(plugin=plugin, name=name, path=path)
        self._plugins[name] = instance
        return instance

    def get(self, name: str) -> PluginInstance | None:
        """Retrieve a loaded plugin by name."""
        return self._plugins.get(name)

    def unload(self, name: str) -> bool:
        """Unload a plugin, freeing its resources."""
        if name in self._plugins:
            del self._plugins[name]
            return True
        return False

    def list_loaded(self) -> list[dict]:
        """List all currently loaded plugins."""
        return [
            {
                "name": inst.name,
                "path": inst.path,
                "type": "instrument" if inst.is_instrument else "effect",
                "parameter_count": len(inst.plugin.parameters),
            }
            for inst in self._plugins.values()
        ]

    def process(
        self,
        plugin_name: str,
        audio: np.ndarray,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """Process audio through a loaded effect plugin."""
        inst = self._plugins.get(plugin_name)
        if inst is None:
            raise ValueError(f"Plugin '{plugin_name}' is not loaded")
        return inst.process_audio(audio, sample_rate)

    def scan_directory(self, directory: str) -> list[dict]:
        """Scan a directory for VST3/AU/CLAP plugin files (non-recursive)."""
        from pathlib import Path as P
        import os

        results = []
        dir_path = P(os.path.expanduser(directory))
        if not dir_path.exists():
            return results

        extensions = {".vst3", ".component", ".clap"}
        for entry in dir_path.iterdir():
            if entry.suffix.lower() in extensions:
                results.append({
                    "name": entry.stem,
                    "path": str(entry),
                    "format": entry.suffix.lstrip(".").upper(),
                    "size_mb": round(sum(
                        f.stat().st_size for f in entry.rglob("*") if f.is_file()
                    ) / 1024 / 1024, 2) if entry.is_dir() else round(
                        entry.stat().st_size / 1024 / 1024, 2
                    ),
                })

        return sorted(results, key=lambda x: x["name"].lower())

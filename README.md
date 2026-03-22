# Audio Plugin MCP Server

A local MCP server that gives Claude the ability to interact with VST3/AU audio plugins programmatically. Load plugins, inspect parameters, process audio, render MIDI through instruments, analyze spectral content, compare outputs, generate presets, and compile JUCE projects — all from within a Claude conversation.

## Installation

### Prerequisites

You need Python 3.10 or later and `pip` installed on your system. On Windows, ensure you're using native Windows Python (not WSL — VST3 plugins cannot load under WSL).

### Step 1: Clone or download this directory

Place the `mcp-audio-plugin-server` folder wherever you keep your MCP servers.

### Step 2: Install dependencies

```bash
cd mcp-audio-plugin-server
pip install -r requirements.txt
```

If you also want DawDreamer for advanced MIDI rendering (optional):
```bash
pip install dawdreamer>=0.8.3
```

### Step 3: Configure plugin paths

Edit `config.yaml` to match your system. The defaults work for a standard Windows 11 installation with VST3 plugins in the standard directory (`C:\Program Files\Common Files\VST3`).

### Step 4: Add to Claude Desktop

Add this block to your `claude_desktop_config.json` (typically at `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "audio-plugin-server": {
      "command": "python",
      "args": ["C:/path/to/mcp-audio-plugin-server/server.py"],
      "env": {}
    }
  }
}
```

Replace the path with the actual location of `server.py` on your machine.

### Step 5: Restart Claude Desktop

The server will start automatically when Claude Desktop launches. Verify by asking Claude to run the `health_check` tool.

## Available Tools (29)

### Plugin Interaction (5 tools)
| Tool | Purpose |
|------|---------|
| `health_check` | Verify server status, dependencies, and configuration |
| `scan_plugins` | Find VST3/AU/CLAP plugins in configured directories |
| `load_plugin` | Load a plugin from a file path, inspect its type and parameters |
| `list_plugin_parameters` | Get full parameter list with values, ranges, and metadata |
| `set_plugin_parameters` | Set one or more parameters on a loaded plugin |

### Audio Processing (4 tools)
| Tool | Purpose |
|------|---------|
| `render_audio` | Process audio through an effect plugin (file input or generated test signals) |
| `render_midi_through_plugin` | Render MIDI notes/chords/scales through an instrument plugin |
| `create_chain` | Build a multi-plugin serial chain and render audio through it in one call |
| `batch_render` | Render with multiple parameter sets for A/B testing and dataset generation |

### Audio File Operations (3 tools)
| Tool | Purpose |
|------|---------|
| `convert_audio` | Convert between formats, sample rates, bit depths, and mono/stereo |
| `trim_audio` | Cut, fade in/out, normalize, or split audio at silence boundaries |
| `concatenate_audio` | Join files end-to-end (with optional crossfade) or layer/mix them together |

### Analysis & Measurement (6 tools)
| Tool | Purpose |
|------|---------|
| `analyze_audio` | Spectral analysis, loudness (LUFS), dynamics, pitch, rhythm, features |
| `compare_audio` | Compare two audio files (spectral similarity, loudness diff, correlation) |
| `extract_impulse_response` | Send impulse through a plugin, capture IR, estimate RT60 |
| `measure_thd` | Total Harmonic Distortion with per-harmonic breakdown |
| `detect_aliasing` | Check for spectral folding artifacts with severity rating |
| `measure_frequency_response` | Sine sweep to measure magnitude response across the spectrum |

### Preset & State Management (5 tools)
| Tool | Purpose |
|------|---------|
| `generate_preset` | Iteratively optimize parameters toward a described target sound |
| `save_plugin_state` | Export all parameter values to a JSON file |
| `load_plugin_state` | Restore parameter values from a saved state file |
| `export_preset_bank` | Save multiple named presets as a bank file |
| `interpolate_presets` | Morph between two parameter sets by a blend factor |
| `randomize_parameters` | Generate random but musically sensible parameter values |

### MIDI Utilities (2 tools)
| Tool | Purpose |
|------|---------|
| `generate_midi_file` | Create MIDI files from note descriptions (name, velocity, beat timing) |
| `analyze_midi` | Parse MIDI files — note count, range, velocity stats, channels, duration |

### Monitoring & Testing (3 tools)
| Tool | Purpose |
|------|---------|
| `monitor_cpu_usage` | Benchmark plugin CPU at various buffer sizes with real-time ratio |
| `stress_test_plugin` | 11-point edge case battery (silence, max level, DC, denormals, extreme params) |
| `compile_juce_project` | Compile a JUCE plugin project from source via CMake |

### Plugin Info (1 tool)
| Tool | Purpose |
|------|---------|
| `get_plugin_latency` | Report latency in samples and milliseconds |

## Example Workflows

### Load a plugin and explore its parameters
> "Scan my VST3 folder, then load FabFilter Pro-Q 3 and list all its parameters."

### Process audio through a chain of effects
> "Load my compressor and EQ plugins, create a chain with the compressor first then EQ, and process my drum loop through it."

### Measure plugin quality
> "Load my distortion plugin, measure the THD at 1kHz, check for aliasing, and plot the frequency response."

### Generate and compare presets
> "Save the current state of my reverb, randomize the parameters, render my vocal through both, and compare the outputs."

### Stress test before release
> "Load my new plugin, run the stress test, then benchmark CPU usage at 64, 128, and 256 sample buffer sizes."

### Create a MIDI file and render through a synth
> "Generate a MIDI file with a C minor chord progression (Cm, Fm, Gm, Cm) at 90 BPM, then render it through my loaded synth plugin."

## Built With

Python, FastMCP, Pedalboard (Spotify), librosa, pyloudnorm, mido, scipy. Runs locally on your machine — your plugins, your audio, your hardware.

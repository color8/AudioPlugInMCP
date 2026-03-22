"""MIDI utilities for note generation, chord construction, and file I/O.

Provides convenience functions for generating MIDI data in the formats
expected by both Pedalboard (list of mido Messages) and DawDreamer
(add_midi_note calls with start_time/duration).
"""

from __future__ import annotations

from dataclasses import dataclass

import mido


# Standard MIDI note name mapping
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class MidiNote:
    """A single MIDI note with timing information."""
    note: int           # MIDI note number 0-127
    velocity: int       # Velocity 1-127
    start_time: float   # Start time in seconds
    duration: float     # Duration in seconds
    channel: int = 0    # MIDI channel 0-15

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def name(self) -> str:
        octave = (self.note // 12) - 1
        name = NOTE_NAMES[self.note % 12]
        return f"{name}{octave}"


def note_name_to_midi(name: str) -> int:
    """Convert a note name like 'C4', 'F#3', 'Bb5' to MIDI number.

    Middle C (C4) = 60. Supports sharps (#) and flats (b).
    """
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"Invalid note name: '{name}'")

    # Parse note letter and accidental
    letter = name[0].upper()
    rest = name[1:]

    accidental = 0
    while rest and rest[0] in ("#", "b"):
        if rest[0] == "#":
            accidental += 1
        else:
            accidental -= 1
        rest = rest[1:]

    octave = int(rest)
    base_notes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    if letter not in base_notes:
        raise ValueError(f"Invalid note letter: '{letter}'")

    return (octave + 1) * 12 + base_notes[letter] + accidental


def make_chord(
    root: int | str,
    chord_type: str = "major",
    velocity: int = 100,
    start_time: float = 0.0,
    duration: float = 1.0,
) -> list[MidiNote]:
    """Generate a chord as a list of MidiNotes.

    Args:
        root: Root note as MIDI number or name (e.g., 60 or 'C4').
        chord_type: One of 'major', 'minor', 'dim', 'aug', 'sus2', 'sus4',
                    'major7', 'minor7', 'dom7'.
        velocity: MIDI velocity for all notes.
        start_time: When the chord starts (seconds).
        duration: How long the chord lasts (seconds).

    Returns:
        List of MidiNote objects forming the chord.
    """
    if isinstance(root, str):
        root = note_name_to_midi(root)

    intervals = {
        "major":  [0, 4, 7],
        "minor":  [0, 3, 7],
        "dim":    [0, 3, 6],
        "aug":    [0, 4, 8],
        "sus2":   [0, 2, 7],
        "sus4":   [0, 5, 7],
        "major7": [0, 4, 7, 11],
        "minor7": [0, 3, 7, 10],
        "dom7":   [0, 4, 7, 10],
    }

    if chord_type not in intervals:
        raise ValueError(
            f"Unknown chord type '{chord_type}'. Available: {list(intervals.keys())}"
        )

    return [
        MidiNote(
            note=min(root + i, 127),
            velocity=velocity,
            start_time=start_time,
            duration=duration,
        )
        for i in intervals[chord_type]
    ]


def make_scale(
    root: int | str,
    scale_type: str = "major",
    velocity: int = 100,
    note_duration: float = 0.5,
    start_time: float = 0.0,
) -> list[MidiNote]:
    """Generate ascending scale notes with sequential timing."""
    if isinstance(root, str):
        root = note_name_to_midi(root)

    scales = {
        "major":     [0, 2, 4, 5, 7, 9, 11, 12],
        "minor":     [0, 2, 3, 5, 7, 8, 10, 12],
        "pentatonic": [0, 2, 4, 7, 9, 12],
        "blues":     [0, 3, 5, 6, 7, 10, 12],
        "chromatic":  list(range(13)),
    }

    if scale_type not in scales:
        raise ValueError(f"Unknown scale '{scale_type}'. Available: {list(scales.keys())}")

    return [
        MidiNote(
            note=min(root + interval, 127),
            velocity=velocity,
            start_time=start_time + i * note_duration,
            duration=note_duration * 0.9,  # Slight gap between notes
        )
        for i, interval in enumerate(scales[scale_type])
    ]


def notes_to_mido_messages(notes: list[MidiNote]) -> list[mido.Message]:
    """Convert MidiNotes to mido Messages for Pedalboard.

    Pedalboard expects a flat list of mido.Message objects where
    the `time` attribute is the absolute time in seconds.
    """
    messages = []
    for note in notes:
        messages.append(mido.Message(
            "note_on", note=note.note, velocity=note.velocity,
            channel=note.channel, time=note.start_time,
        ))
        messages.append(mido.Message(
            "note_off", note=note.note, velocity=0,
            channel=note.channel, time=note.end_time,
        ))
    # Sort by time to ensure correct ordering
    messages.sort(key=lambda m: m.time)
    return messages


def notes_to_raw_midi(notes: list[MidiNote]) -> list[tuple[list[int], float]]:
    """Convert MidiNotes to raw MIDI byte tuples for Pedalboard.

    Alternative to mido — returns list of (bytes, timestamp) tuples.
    """
    events = []
    for note in notes:
        events.append(([0x90 | note.channel, note.note, note.velocity], note.start_time))
        events.append(([0x80 | note.channel, note.note, 0], note.end_time))
    events.sort(key=lambda e: e[1])
    return events


def load_midi_file(path: str) -> list[MidiNote]:
    """Load a MIDI file and return a list of MidiNotes with absolute timing."""
    mid = mido.MidiFile(path)
    notes = []

    for track in mid.tracks:
        abs_time = 0.0
        pending: dict[tuple[int, int], float] = {}  # (channel, note) -> start_time

        for msg in track:
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, 500000)

            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = abs_time
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in pending:
                    start = pending.pop(key)
                    notes.append(MidiNote(
                        note=msg.note,
                        velocity=msg.velocity if msg.type == "note_on" else 80,
                        start_time=start,
                        duration=abs_time - start,
                        channel=msg.channel,
                    ))

    notes.sort(key=lambda n: n.start_time)
    return notes

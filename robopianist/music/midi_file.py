# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for working with MIDI files.

TODO(kevin):
- Figure out a better way for supporting fingering information. It would be nice if a
a field were added to the NoteSequence proto but unclear if the Magenta team is willing
to do that.
- Replace all conversion functions with those in pretty_midi. No point in re-inventing
the wheel.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from note_seq import NoteSequence, midi_io, midi_synth, music_pb2, sequences_lib
from note_seq import constants as ns_constants

from robopianist import SF2_PATH
from robopianist.music import audio
from robopianist.music import constants as consts
from robopianist.music.piano_roll import sequence_to_pianoroll


def note_name_to_midi_number(name: str) -> int:
    """Returns the MIDI pitch number for the given note name.

    Args:
        name: Note name, e.g. "C#5".

    Returns:
        MIDI pitch number, e.g. 73.
    """
    return consts.NOTE_NAME_TO_MIDI_NUMBER[name]


def midi_number_to_note_name(number: int) -> str:
    """Returns the note name for the given MIDI pitch number.

    Args:
        number: MIDI pitch number, e.g. 73.

    Returns:
        Note name, e.g. "C#5".
    """
    return consts.MIDI_NUMBER_TO_NOTE_NAME[number]


def key_number_to_midi_number(key_number: int) -> int:
    """Returns the MIDI pitch number for the given piano key number.

    Args:
        key_number: Key number, e.g. 0 for A0, 87 for C8.

    Returns:
        MIDI pitch number.
    """
    if not 0 <= key_number < consts.NUM_KEYS:
        raise ValueError(
            f"Key number should be in [0 {consts.NUM_KEYS}], got {key_number}."
        )
    return key_number + consts.MIN_MIDI_PITCH_PIANO


def midi_number_to_key_number(midi_number: int) -> int:
    """Returns the piano key number for the given MIDI pitch number.

    Args:
        midi_number: MIDI pitch number.

    Returns:
        Key number, e.g. 0 for A0, 87 for C8.
    """
    if not consts.MIN_MIDI_PITCH_PIANO <= midi_number <= consts.MAX_MIDI_PITCH_PIANO:
        raise ValueError(
            f"MIDI pitch number should be in [{consts.MIN_MIDI_PITCH_PIANO}, "
            f"{consts.MAX_MIDI_PITCH_PIANO}], got {midi_number}."
        )
    return midi_number - consts.MIN_MIDI_PITCH_PIANO


def key_number_to_note_name(key_number: int) -> str:
    """Returns the note name for the given piano key number.

    Args:
        key_number: Key number, e.g. 0 for A0, 87 for C8.

    Returns:
        Note name, e.g. "C#5".
    """
    return consts.KEY_NUMBER_TO_NOTE_NAME[key_number]


def note_name_to_key_number(note_name: str) -> int:
    """Returns the piano key number for the given note name.

    Args:
        note_name: Note name, e.g. "C#5".

    Returns:
        Key number, e.g. 0 for A0, 87 for C8.
    """
    return consts.NOTE_NAME_TO_KEY_NUMBER[note_name]


@dataclass(frozen=True)
class PianoNote:
    """A container for a piano note.

    Attributes:
        number: MIDI key number.
        velocity: How hard the key was struck.
        key: Piano key corresponding to the note.
        name: Note name, in scientific pitch notation.
        fingering: Optional fingering for the note. Right hand fingers are numbered 0
            to 4, left hand 5 to 9, both starting from the thumb and ending at the
            pinky. -1 means no fingering information is available.
    """

    number: int
    velocity: int
    key: int
    name: str
    fingering: int = -1

    @staticmethod
    def create(number: int, velocity: int, fingering: int = -1) -> "PianoNote":
        """Creates a PianoNote from a MIDI pitch number and velocity."""
        if (
            not ns_constants.MIN_MIDI_VELOCITY
            <= velocity
            <= ns_constants.MAX_MIDI_VELOCITY
        ):
            raise ValueError(
                f"Velocity should be in [{ns_constants.MIN_MIDI_VELOCITY}, "
                f"{ns_constants.MAX_MIDI_VELOCITY}], got {velocity}."
            )
        if not consts.MIN_MIDI_PITCH_PIANO <= number <= consts.MAX_MIDI_PITCH_PIANO:
            raise ValueError(
                f"MIDI pitch number should be in [{consts.MIN_MIDI_PITCH_PIANO}, "
                f"{consts.MAX_MIDI_PITCH_PIANO}], got {number}."
            )
        return PianoNote(
            number=number,
            velocity=velocity,
            key=midi_number_to_key_number(number),
            name=midi_number_to_note_name(number),
            fingering=fingering,
        )


@dataclass(frozen=True)
class MidiFile:
    """Abstraction for working with MIDI files."""

    seq: NoteSequence

    # Factory methods.

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> "MidiFile":
        """Loads a MIDI file (either .mid or .proto) from disk."""
        filename = Path(filename)
        if filename.suffix == ".mid":
            try:
                seq = midi_io.midi_file_to_note_sequence(midi_file=filename)
            except midi_io.MIDIConversionError:
                raise RuntimeError(f"Could not parse MIDI file {filename}.")
        elif filename.suffix == ".proto":
            seq = music_pb2.NoteSequence()
            with open(filename, "rb") as f:
                seq.ParseFromString(f.read())
        else:
            raise ValueError(f"Unsupported file extension {filename.suffix}.")
        return cls(seq=seq)

    def save(self, filename: Union[str, Path]) -> None:
        """Save the MIDI file (either .mid or .proto) to disk."""
        filename = Path(filename)
        if filename.suffix == ".mid":
            midi_io.note_sequence_to_midi_file(self.seq, filename)
        elif filename.suffix == ".proto":
            with open(filename, "wb") as f:
                f.write(self.seq.SerializeToString())
        else:
            raise ValueError(f"Unsupported file extension {filename.suffix}.")

    # Main methods.

    def stretch(self, factor: float) -> "MidiFile":
        """Stretch the MIDI file by the given factor.

        Values greater than 1 stretch the sequence (i.e., make it slower), values less
        than 1 compress it (i.e., make it faster). Zero and negative values are not
        allowed.
        """
        if factor <= 0:
            raise ValueError("factor must be positive.")
        # NOTE: stretch_note_sequence is a no-op if factor == 1.
        return MidiFile(seq=sequences_lib.stretch_note_sequence(self.seq, factor))

    def transpose(self, amount: int, transpose_chords: bool = True) -> "MidiFile":
        """Transpose the MIDI file by the given amount of semitones.

        Positive values transpose up, negative values transpose down. Out of range notes
        (i.e., outside the min and max piano range) are removed from the sequence.
        """
        seq, _ = sequences_lib.transpose_note_sequence(
            self.seq,
            amount=amount,
            min_allowed_pitch=consts.MIN_MIDI_PITCH_PIANO,
            max_allowed_pitch=consts.MAX_MIDI_PITCH_PIANO,
            transpose_chords=transpose_chords,
        )
        return MidiFile(seq=seq)

    def synthesize(self, sampling_rate: int = consts.SAMPLING_RATE) -> np.ndarray:
        """Synthesize the MIDI file into a waveform using FluidSynth."""
        return midi_synth.fluidsynth(
            self.seq, sample_rate=sampling_rate, sf2_path=str(SF2_PATH)
        )

    def play(self, sampling_rate: int = consts.SAMPLING_RATE) -> None:
        """Play the MIDI file using FluidSynth and PyAudio."""
        waveform_float = self.synthesize()
        normalizer = float(np.iinfo(np.int16).max)
        waveform = np.array(np.asarray(waveform_float) * normalizer, dtype=np.int16)
        audio.play_sound(waveform, sampling_rate=sampling_rate)

    def has_fingering(self) -> bool:
        """Returns whether the MIDI file has fingering information."""
        # A MIDI file has fingering information if it has more than one unique part
        # number (because the part field is 0 by default) and at least one of those
        # part numbers is non-zero.
        fingerings = set()
        for note in self.seq.notes:
            fingerings.add(note.part)
        non_zero_fingerings = [f for f in fingerings if f != 0]
        return len(fingerings) > 1 and len(non_zero_fingerings) > 0

    # Accessors.

    @property
    def duration(self) -> float:
        """Returns the duration of the MIDI file in seconds."""
        return self.seq.total_time

    @property
    def n_notes(self) -> int:
        """Returns the number of notes in the MIDI file."""
        return len(self.seq.notes)

    @property
    def title(self) -> str:
        """Returns the name of the MIDI file. Empty string if not specified."""
        return self.seq.sequence_metadata.title

    @property
    def artist(self) -> str:
        """Returns the artist of the MIDI file. Empty string if not specified."""
        return self.seq.sequence_metadata.artist


@dataclass
class NoteTrajectory:
    """A time series representation of a MIDI file.

    Attributes:
        dt: The discretization time step in seconds.
        notes: A list of lists of PianoNotes. The outer list is indexed by time step,
            and the inner list contains all the notes that are active at that time step.
        sustains: A list of integers. The i-th element indicates whether the sustain
            pedal is active at the i-th time step.
    """

    dt: float
    notes: List[List[PianoNote]]
    sustains: List[int]

    def __post_init__(self) -> None:
        """Validates the attributes."""
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if len(self.notes) != len(self.sustains):
            raise ValueError("notes and sustains must have the same length.")

    @classmethod
    def from_midi(cls, midi: MidiFile, dt: float) -> "NoteTrajectory":
        """Constructs a NoteTrajectory from a MIDI file."""
        notes, sustains = NoteTrajectory.seq_to_trajectory(midi.seq, dt)
        return cls(dt=dt, notes=notes, sustains=sustains)

    @staticmethod
    def seq_to_trajectory(
        seq: NoteSequence, dt: float
    ) -> Tuple[List[List[PianoNote]], List[int]]:
        """Converts a NoteSequence into a time series representation."""
        # Convert the note sequence into a piano roll.
        piano_roll = sequence_to_pianoroll(
            seq,
            frames_per_second=1 / dt,
            min_pitch=consts.MIN_MIDI_PITCH,
            max_pitch=consts.MAX_MIDI_PITCH,
            onset_window=0,
        )

        # Find the set of active notes at each timestep.
        notes: List[List[PianoNote]] = []
        for t, timestep in enumerate(piano_roll.active_velocities):
            notes_in_timestep: List[PianoNote] = []
            for index in np.nonzero(timestep)[0]:
                if (
                    t > 0
                    and piano_roll.active_velocities[t - 1][index]
                    and piano_roll.onset_velocities[t][index]
                ):
                    # This is to disambiguate notes that are sustained for multiple
                    # timesteps vs notes that are played consecutively over multiple
                    # timesteps.
                    continue
                velocity = int(round(timestep[index] * consts.MAX_VELOCITY))
                fingering = int(piano_roll.fingerings[t, index])
                notes_in_timestep.append(PianoNote.create(index, velocity, fingering))
            notes.append(notes_in_timestep)

        # Find the sustain pedal state at each timestep.
        sustains: List[int] = []
        prev_sustain = 0
        for timestep in piano_roll.control_changes:
            event = timestep[consts.SUSTAIN_PEDAL_CC_NUMBER]
            if 1 <= event <= consts.SUSTAIN_PEDAL_CC_NUMBER:
                sustain = 0
            elif consts.SUSTAIN_PEDAL_CC_NUMBER + 1 <= event <= consts.MAX_CC_VALUE + 1:
                sustain = 1
            else:
                sustain = prev_sustain
            sustains.append(sustain)
            prev_sustain = sustain

        return notes, sustains

    def __len__(self) -> int:
        return len(self.notes)

    def trim_silence(self) -> "NoteTrajectory":
        """Removes any leading or trailing silence from the note trajectory.

        This method modifies the note trajectory in place.
        """
        # Continue removing from the front until we find a non-empty timestep.
        while len(self.notes) > 0 and len(self.notes[0]) == 0:
            self.notes.pop(0)
            self.sustains.pop(0)

        # Continue removing from the back until we find a non-empty timestep.
        while len(self.notes) > 0 and len(self.notes[-1]) == 0:
            self.notes.pop(-1)
            self.sustains.pop(-1)

        return self

    def add_initial_buffer_time(self, initial_buffer_time: float) -> "NoteTrajectory":
        """Adds artificial silence to the start of the note trajectory.

        This method modifies the note trajectory in place.
        """
        if initial_buffer_time < 0.0:
            raise ValueError("initial_buffer_time must be non-negative.")

        for _ in range(int(round(initial_buffer_time / self.dt))):
            self.notes.insert(0, [])
            self.sustains.insert(0, 0)

        return self

    def to_piano_roll(self) -> np.ndarray:
        """Returns a piano roll representation of the note trajectory.

        The piano roll is a 2D array of shape (num_timesteps, num_pitches). Each row is
        a timestep, and each column is a pitch. The value at each cell is 1 if the note
        is active at that timestep, and 0 otherwise.
        """
        frames = np.zeros((len(self.notes), consts.MAX_MIDI_PITCH), dtype=np.int32)
        for t, timestep in enumerate(self.notes):
            for note in timestep:
                frames[t, note.number] = 1
        return frames

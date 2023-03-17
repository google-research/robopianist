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

"""Common variations for the suite."""

from typing import Sequence

import numpy as np
from dm_control.composer import variation
from dm_control.composer.variation import distributions

from robopianist import music
from robopianist.music import constants, midi_file


class MidiSelect(variation.Variation):
    """Randomly select a MIDI file from the registry."""

    def __init__(self, midi_names: Sequence[str] = []) -> None:
        """Initializes the variation.

        Args:
            midi_names: A sequence of MIDI names to select from. Must be valid keys that
                can be loaded by `robopianist.music.load`.
        """
        self._midi_names = midi_names
        self._dist = distributions.UniformChoice(midi_names)

    def __call__(
        self, initial_value=None, current_value=None, random_state=None
    ) -> midi_file.MidiFile:
        del initial_value, current_value  # Unused.
        random = random_state or np.random
        midi_key: str = self._dist(random_state=random)
        return music.load(midi_key)


class MidiTemporalStretch(variation.Variation):
    """Randomly apply a temporal stretch to a MIDI file."""

    def __init__(
        self,
        prob: float,
        stretch_range: float,
    ) -> None:
        """Initializes the variation.

        Args:
            prob: A float specifying the probability of applying a temporal stretch.
            stretch_range: Range specifying the bounds of the uniform distribution
                from which the multiplicative stretch factor is sampled from (i.e.,
                [1 - stretch_range, 1 + stretch_range]).
        """
        self._prob = prob
        self._dist = distributions.Uniform(-stretch_range, stretch_range)

    def __call__(
        self, initial_value=None, current_value=None, random_state=None
    ) -> midi_file.MidiFile:
        del current_value  # Unused.
        random = random_state or np.random
        if random.uniform(0.0, 1.0) > self._prob:
            if initial_value is None or not isinstance(
                initial_value, midi_file.MidiFile
            ):
                raise ValueError(
                    "Expected `initial_value` to be provided and be a midi_file.MidiFile."
                )
            return initial_value
        stretch_factor = 1.0 + self._dist(random_state=random)
        return initial_value.stretch(stretch_factor)


class MidiPitchShift(variation.Variation):
    """Randomly apply a pitch shift to a MIDI file."""

    def __init__(
        self,
        prob: float,
        shift_range: int,
    ) -> None:
        """Initializes the variation.

        Args:
            prob: A float specifying the probability of applying a pitch shift.
            shift_range: Range specifying the maximum absolute value of the uniform
                distribution from which the pitch shift, in semitones, is sampled from.
                This value will get truncated to the maximum number of semitones that
                can be shifted without exceeding the piano's range.
        """
        self._prob = prob
        if not isinstance(shift_range, int):
            raise ValueError("`shift_range` must be an integer.")
        self._shift_range = shift_range

    def __call__(
        self, initial_value=None, current_value=None, random_state=None
    ) -> midi_file.MidiFile:
        del current_value  # Unused.
        random = random_state or np.random
        if random.uniform(0.0, 1.0) > self._prob:
            if initial_value is None or not isinstance(
                initial_value, midi_file.MidiFile
            ):
                raise ValueError(
                    "Expected `initial_value` to be provided and be a midi_file.MidiFile."
                )
            return initial_value

        if self._shift_range == 0:
            return initial_value

        # Ensure that the pitch shift won't exceed the piano's range.
        pitches = [note.pitch for note in initial_value.seq.notes]
        min_pitch, max_pitch = min(pitches), max(pitches)
        low = max(constants.MIN_MIDI_PITCH_PIANO - min_pitch, -self._shift_range)
        high = min(constants.MAX_MIDI_PITCH_PIANO - max_pitch, self._shift_range)

        shift = random.randint(low, high + 1)
        if shift == 0:
            return initial_value
        return initial_value.transpose(shift)


class MidiOctaveShift(variation.Variation):
    """Shift the pitch of a MIDI file in octaves."""

    def __init__(
        self,
        prob: float,
        octave_range: int,
    ) -> None:
        """Initializes the variation.

        Args:
            prob: A float specifying the probability of applying a pitch shift.
            octave_range: Range specifying the maximum absolute value of the uniform
                distribution from which the octave shift is sampled from. This value
                will get truncated to the maximum number of octaves that can be
                shifted without exceeding the piano's range.
        """
        self._prob = prob
        if not isinstance(octave_range, int):
            raise ValueError("`octave_range` must be an integer.")
        self._octave_range = octave_range

    def __call__(
        self, initial_value=None, current_value=None, random_state=None
    ) -> midi_file.MidiFile:
        del current_value  # Unused.
        random = random_state or np.random
        if random.uniform(0.0, 1.0) > self._prob:
            if initial_value is None or not isinstance(
                initial_value, midi_file.MidiFile
            ):
                raise ValueError(
                    "Expected `initial_value` to be provided and be a midi_file.MidiFile."
                )
            return initial_value

        if self._octave_range == 0:
            return initial_value

        # Ensure that the octave shift won't exceed the piano's range.
        pitches = [note.pitch for note in initial_value.seq.notes]
        min_pitch, max_pitch = min(pitches), max(pitches)
        low = max(constants.MIN_MIDI_PITCH_PIANO - min_pitch, -self._octave_range * 12)
        high = min(constants.MAX_MIDI_PITCH_PIANO - max_pitch, self._octave_range * 12)

        shift = random.randint(low // 12, high // 12 + 1)
        if shift == 0:
            return initial_value
        return initial_value.transpose(shift * 12)

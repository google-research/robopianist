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

"""Library for synthesizing music from MIDI files."""

from pathlib import Path
from typing import Sequence

import fluidsynth
import numpy as np

from robopianist import SF2_PATH
from robopianist.music import constants as consts
from robopianist.music import midi_message
from robopianist.music.constants import SAMPLING_RATE

_PROGRAM = 0  # Acoustic Grand Piano
_CHANNEL = 0
_BANK = 0


def _validate_note(note: int) -> None:
    assert consts.MIN_MIDI_PITCH <= note <= consts.MAX_MIDI_PITCH


def _validate_velocity(velocity: int) -> None:
    assert consts.MIN_VELOCITY <= velocity <= consts.MAX_VELOCITY


class Synthesizer:
    """FluidSynth-based synthesizer."""

    def __init__(
        self,
        soundfont_path: Path = SF2_PATH,
        sample_rate: int = SAMPLING_RATE,
    ) -> None:
        self._soundfont_path = soundfont_path
        self._sample_rate = sample_rate
        self._muted: bool = False
        self._sustained: bool = False

        # Initialize FluidSynth.
        self._synth = fluidsynth.Synth(samplerate=float(sample_rate))
        soundfont_id = self._synth.sfload(str(soundfont_path))
        self._synth.program_select(_CHANNEL, soundfont_id, _BANK, _PROGRAM)

    def start(self) -> None:
        self._synth.start()

    def stop(self) -> None:
        self._synth.delete()

    def mute(self, value: bool) -> None:
        self._muted = value
        if value:
            self.all_sounds_off()

    def all_sounds_off(self) -> None:
        self._synth.all_sounds_off(_CHANNEL)

    def all_notes_off(self) -> None:
        self._synth.all_notes_off(_CHANNEL)

    def note_on(self, note: int, velocity: int) -> None:
        if not self._muted:
            _validate_note(note)
            _validate_velocity(velocity)
            self._synth.noteon(_CHANNEL, note, velocity)

    def note_off(self, note: int) -> None:
        if not self._muted:
            _validate_note(note)
            self._synth.noteoff(_CHANNEL, note)

    def sustain_on(self) -> None:
        if not self._muted:
            self._synth.cc(
                _CHANNEL, consts.SUSTAIN_PEDAL_CC_NUMBER, consts.MAX_CC_VALUE
            )
            self._sustained = True

    def sustain_off(self) -> None:
        if not self._muted:
            self._synth.cc(
                _CHANNEL, consts.SUSTAIN_PEDAL_CC_NUMBER, consts.MIN_CC_VALUE
            )
            self._sustained = False

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def sustained(self) -> bool:
        return self._sustained

    def get_samples(
        self,
        event_list: Sequence[midi_message.MidiMessage],
    ) -> np.ndarray:
        """Synthesize a list of MIDI events into a waveform."""
        current_time = event_list[0].time

        # Convert absolute seconds to relative seconds.
        next_event_times = [e.time for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event.time = end - event.time

        # Include 1 second of silence at the end.
        event_list[-1].time = 1.0

        total_time = current_time + np.sum([e.time for e in event_list])
        synthesized = np.zeros(int(np.ceil(self._sample_rate * total_time)))
        for event in event_list:
            if isinstance(event, midi_message.NoteOn):
                self.note_on(event.note, event.velocity)
            elif isinstance(event, midi_message.NoteOff):
                self.note_off(event.note)
            elif isinstance(event, midi_message.SustainOn):
                self.sustain_on()
            elif isinstance(event, midi_message.SustainOff):
                self.sustain_off()
            else:
                raise ValueError(f"Unknown event type: {event}")
            current_sample = int(self._sample_rate * current_time)
            end = int(self._sample_rate * (current_time + event.time))
            samples = self._synth.get_samples(end - current_sample)[::2]
            synthesized[current_sample:end] += samples
            current_time += event.time
        waveform_float = synthesized / np.abs(synthesized).max()

        # Convert to 16-bit ints.
        normalizer = float(np.iinfo(np.int16).max)
        return np.array(np.asarray(waveform_float) * normalizer, dtype=np.int16)

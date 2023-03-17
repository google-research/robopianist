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

"""Stripped down MIDI message class."""

import enum
from dataclasses import dataclass
from typing import Union

from robopianist.music import constants as consts


@enum.unique
class EventType(enum.Enum):
    """The type of a MIDI event."""

    NOTE_ON = enum.auto()
    NOTE_OFF = enum.auto()
    SUSTAIN_ON = enum.auto()
    SUSTAIN_OFF = enum.auto()


@dataclass
class NoteOn:
    """A note-on MIDI message."""

    note: int
    velocity: int
    time: float

    def __post_init__(self) -> None:
        assert consts.MIN_MIDI_PITCH <= self.note <= consts.MAX_MIDI_PITCH
        assert consts.MIN_VELOCITY <= self.velocity <= consts.MAX_VELOCITY
        assert self.time >= 0

    @property
    def event_type(self) -> EventType:
        return EventType.NOTE_ON


@dataclass
class NoteOff:
    """A note-off MIDI message."""

    note: int
    time: float

    def __post_init__(self) -> None:
        assert consts.MIN_MIDI_PITCH <= self.note <= consts.MAX_MIDI_PITCH
        assert self.time >= 0

    @property
    def event_type(self) -> EventType:
        return EventType.NOTE_OFF


@dataclass
class _ControlChange:
    """A control-change MIDI message."""

    control: int
    value: int
    time: float

    def __post_init__(self) -> None:
        assert consts.MIN_CC_VALUE <= self.control <= consts.MAX_CC_VALUE
        assert consts.MIN_CC_VALUE <= self.value <= consts.MAX_CC_VALUE
        assert self.time >= 0


class SustainOn(_ControlChange):
    """A sustain-on MIDI message."""

    def __init__(self, time: float) -> None:
        super().__init__(
            consts.SUSTAIN_PEDAL_CC_NUMBER,
            consts.SUSTAIN_PEDAL_CC_NUMBER,
            time,
        )

    @property
    def event_type(self) -> EventType:
        return EventType.SUSTAIN_ON


class SustainOff(_ControlChange):
    """A sustain-off MIDI message."""

    def __init__(self, time: float) -> None:
        super().__init__(
            consts.SUSTAIN_PEDAL_CC_NUMBER,
            0,
            time,
        )

    @property
    def event_type(self) -> EventType:
        return EventType.SUSTAIN_OFF


MidiMessage = Union[NoteOn, NoteOff, SustainOn, SustainOff]

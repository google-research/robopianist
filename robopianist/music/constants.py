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

"""Music constants."""

MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127

# MIDI pitch number of the lowest note on the piano (A0).
MIN_MIDI_PITCH_PIANO = 21
# MIDI pitch number of the highest note on the piano (C8).
MAX_MIDI_PITCH_PIANO = 108

# Min and max key numbers on the piano.
MIN_KEY_NUMBER = 0
MAX_KEY_NUMBER = 87
NUM_KEYS = MAX_KEY_NUMBER - MIN_KEY_NUMBER + 1

# Notes in an octave.
NOTES_IN_OCTAVE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
assert len(NOTES_IN_OCTAVE) == 12

# Notes on an 88-key piano from left (A0) to right (C8).
NOTES = []
NOTES.extend(["A0", "A#0", "B0"])
for octave in range(1, 8):
    for note in NOTES_IN_OCTAVE:
        NOTES.append(note + str(octave))
NOTES.append("C8")
assert len(NOTES) == NUM_KEYS

# Mapping for converting between a key number and its note name.
KEY_NUMBER_TO_NOTE_NAME = {i: note for i, note in enumerate(NOTES)}
NOTE_NAME_TO_KEY_NUMBER = {note: i for i, note in enumerate(NOTES)}

# Mapping for converting between a note name and its MIDI pitch number.
MIDI_NUMBER_TO_NOTE_NAME = {i + 21: name for i, name in enumerate(NOTES)}
NOTE_NAME_TO_MIDI_NUMBER = {v: k for k, v in MIDI_NUMBER_TO_NOTE_NAME.items()}

# Sampling frequency of the audio, in Hz.
SAMPLING_RATE = 44100

SUSTAIN_PEDAL_CC_NUMBER = 64
MIN_CC_VALUE = 0
MAX_CC_VALUE = 127

MIN_VELOCITY = 0
MAX_VELOCITY = 127

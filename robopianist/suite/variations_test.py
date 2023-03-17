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

"""Tests for variations.py."""

import numpy as np
from absl.testing import absltest
from note_seq.protobuf import compare

from robopianist.music import ALL, library, midi_file
from robopianist.suite import variations

_SEED = 12345
_NUM_SAMPLES = 100


class MidiSelectTest(absltest.TestCase):
    def test_output_is_midi_file(self) -> None:
        var = variations.MidiSelect(midi_names=ALL)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            midi = var(random_state=random_state)
            self.assertIsInstance(midi, midi_file.MidiFile)


class MidiTemporalStretchTest(absltest.TestCase):
    def assertProtoEquals(self, a, b, msg=None):
        if not compare.ProtoEq(a, b):
            compare.assertProtoEqual(self, a, b, normalize_numbers=True, msg=msg)

    def test_output_is_midi_file(self) -> None:
        original_midi = library.toy()
        var = variations.MidiTemporalStretch(prob=0.1, stretch_range=0.5)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            midi = var(initial_value=original_midi, random_state=random_state)
            self.assertIsInstance(midi, midi_file.MidiFile)

    def test_output_same_if_prob_zero(self) -> None:
        original_midi = library.toy()
        var = variations.MidiTemporalStretch(prob=0.0, stretch_range=0.5)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            new_midi = var(initial_value=original_midi, random_state=random_state)
            self.assertIs(original_midi, new_midi)

    def test_output_different_if_prob_one(self) -> None:
        original_midi = library.toy()
        var = variations.MidiTemporalStretch(prob=1.0, stretch_range=0.5)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            new_midi = var(initial_value=original_midi, random_state=random_state)
            self.assertIsNot(original_midi, new_midi)

    def test_raises_value_error_if_no_initial_value(self) -> None:
        var = variations.MidiTemporalStretch(prob=0.1, stretch_range=0.5)
        random_state = np.random.RandomState(_SEED)
        with self.assertRaises(ValueError):
            var(random_state=random_state)

    def test_raises_value_error_if_wrong_type(self) -> None:
        var = variations.MidiTemporalStretch(prob=0.1, stretch_range=0.5)
        random_state = np.random.RandomState(_SEED)
        with self.assertRaises(ValueError):
            var(initial_value=1, random_state=random_state)

    def test_output_same_if_stretch_range_zero(self) -> None:
        original_midi = library.toy()
        var = variations.MidiTemporalStretch(prob=0.1, stretch_range=0.0)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            new_midi = var(initial_value=original_midi, random_state=random_state)
            self.assertProtoEquals(original_midi.seq, new_midi.seq)
            self.assertEqual(original_midi.duration, new_midi.duration)


class MidiPitchShiftTest(absltest.TestCase):
    def assertProtoEquals(self, a, b, msg=None):
        if not compare.ProtoEq(a, b):
            compare.assertProtoEqual(self, a, b, normalize_numbers=True, msg=msg)

    def test_output_is_midi_file(self) -> None:
        original_midi = library.toy()
        var = variations.MidiPitchShift(prob=0.1, shift_range=1)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            midi = var(initial_value=original_midi, random_state=random_state)
            self.assertIsInstance(midi, midi_file.MidiFile)

    def test_output_same_if_prob_zero(self) -> None:
        original_midi = library.toy()
        var = variations.MidiPitchShift(prob=0.0, shift_range=1)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            new_midi = var(initial_value=original_midi, random_state=random_state)
            self.assertIs(original_midi, new_midi)

    def test_raises_value_error_if_no_initial_value(self) -> None:
        var = variations.MidiPitchShift(prob=0.1, shift_range=1)
        random_state = np.random.RandomState(_SEED)
        with self.assertRaises(ValueError):
            var(random_state=random_state)

    def test_raises_value_error_if_wrong_type(self) -> None:
        var = variations.MidiPitchShift(prob=0.1, shift_range=1)
        random_state = np.random.RandomState(_SEED)
        with self.assertRaises(ValueError):
            var(initial_value=1, random_state=random_state)

    def test_output_same_if_range_zero(self) -> None:
        original_midi = library.toy()
        var = variations.MidiPitchShift(prob=0.1, shift_range=0)
        random_state = np.random.RandomState(_SEED)
        for _ in range(_NUM_SAMPLES):
            new_midi = var(initial_value=original_midi, random_state=random_state)
            self.assertIs(original_midi, new_midi)


if __name__ == "__main__":
    absltest.main()

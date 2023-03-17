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

"""Tests for library.py."""

from absl.testing import absltest, parameterized

from robopianist import music
from robopianist.music import midi_file


class ConstantsTest(parameterized.TestCase):
    def test_constants(self) -> None:
        # Check that all constants are non-empty.
        self.assertNotEmpty(music.ALL)
        self.assertNotEmpty(music.DEBUG_MIDIS)
        self.assertNotEmpty(music.PIG_MIDIS)
        self.assertNotEmpty(music.ETUDE_MIDIS)

        # Check that all = debug + pig.
        self.assertEqual(music.ALL, music.DEBUG_MIDIS + music.PIG_MIDIS)

        # Check that etude is a subset of pig.
        self.assertTrue(set(music.ETUDE_MIDIS).issubset(set(music.PIG_MIDIS)))


class LoadTest(parameterized.TestCase):
    def test_raises_key_error_on_invalid_midi(self) -> None:
        """Test that loading an invalid string MIDI raises a KeyError."""
        with self.assertRaises(KeyError):
            music.load("invalid_midi")

    @parameterized.parameters(*music.ALL)
    def test_midis_in_library(self, midi_name: str) -> None:
        """Test that all midis in the library can be loaded."""
        self.assertIsInstance(music.load(midi_name), midi_file.MidiFile)


if __name__ == "__main__":
    absltest.main()

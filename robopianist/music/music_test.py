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

from robopianist import _PROJECT_ROOT, music
from robopianist.music import midi_file

_PIG_DIR = _PROJECT_ROOT / "robopianist" / "music" / "data" / "pig_single_finger"


@absltest.skipIf(not _PIG_DIR.exists(), "PIG dataset not found.")
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


@absltest.skipIf(not _PIG_DIR.exists(), "PIG dataset not found.")
class LoadTest(parameterized.TestCase):
    def test_raises_key_error_on_invalid_midi(self) -> None:
        """Test that loading an invalid string MIDI raises a KeyError."""
        with self.assertRaises(KeyError):
            music.load("invalid_midi")

    @parameterized.parameters(*music.ALL)
    def test_midis_in_library(self, midi_name: str) -> None:
        """Test that all midis in the library can be loaded."""
        self.assertIsInstance(music.load(midi_name), midi_file.MidiFile)

    @parameterized.parameters(*music.ALL)
    def test_fingering_available_for_all_timesteps(self, midi_name: str) -> None:
        """Test that all midis in the library have fingering annotations for all
        timesteps."""
        midi = music.load(midi_name).trim_silence()
        traj = midi_file.NoteTrajectory.from_midi(midi, dt=0.05)
        for timestep in traj.notes:
            for note in timestep:
                # -1 indicates no fingering annotation. Valid fingering lies in [0, 9].
                self.assertGreater(note.fingering, -1)
                self.assertLess(note.fingering, 10)


if __name__ == "__main__":
    absltest.main()

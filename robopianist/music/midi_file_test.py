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

"""Tests for midi_file.py."""

from absl.testing import absltest, parameterized
from note_seq.protobuf import compare, music_pb2

from robopianist import music
from robopianist.music import midi_file


class MidiFileTest(parameterized.TestCase):
    def assertProtoEquals(self, a, b, msg=None):
        if not compare.ProtoEq(a, b):
            compare.assertProtoEqual(self, a, b, normalize_numbers=True, msg=msg)

    @parameterized.parameters(0.5, 1.0, 2.0)
    def test_temporal_stretch(self, stretch_factor: float) -> None:
        midi = music.load("CMajorScaleTwoHands")
        stretched_midi = midi.stretch(stretch_factor)
        self.assertEqual(stretched_midi.n_notes, midi.n_notes)
        self.assertEqual(stretched_midi.duration, midi.duration * stretch_factor)

    @parameterized.parameters(-1, 0)
    def test_temporal_stretch_raises_value_error(self, stretch_factor: float) -> None:
        midi = music.load("CMajorScaleTwoHands")
        with self.assertRaises(ValueError):
            midi.stretch(stretch_factor)

    def test_temporal_stretch_no_op(self) -> None:
        midi = music.load("CMajorScaleTwoHands")
        stretched_midi = midi.stretch(1.0)
        self.assertProtoEquals(stretched_midi.seq, midi.seq)

    @parameterized.parameters(-2, -1, 0, 1, 2)
    def test_transpose(self, amount: int) -> None:
        midi = music.load("CMajorScaleTwoHands")
        stretched_midi = midi.transpose(amount)
        self.assertEqual(stretched_midi.n_notes, midi.n_notes)
        # TODO(kevin): Check that the notes are actually transposed.

    def test_transpose_no_op(self) -> None:
        midi = music.load("CMajorScaleTwoHands")
        transposed_midi = midi.transpose(0)
        self.assertProtoEquals(transposed_midi.seq, midi.seq)

    def test_trim_silence(self) -> None:
        midi = music.load("TwinkleTwinkleRousseau")
        midi_trimmed = midi.trim_silence()
        self.assertEqual(midi_trimmed.seq.notes[0].start_time, 0.0)


class PianoNoteTest(absltest.TestCase):
    def test_constructor(self) -> None:
        name = "C4"
        number = midi_file.note_name_to_midi_number(name)
        velocity = 100
        note = midi_file.PianoNote.create(number=number, velocity=velocity)
        self.assertEqual(note.number, number)
        self.assertEqual(note.velocity, velocity)
        self.assertEqual(note.name, name)

    def test_raises_value_error_negative_number(self) -> None:
        with self.assertRaises(ValueError):
            midi_file.PianoNote.create(number=-1, velocity=0)

    def test_raises_value_error_large_number(self) -> None:
        with self.assertRaises(ValueError):
            midi_file.PianoNote.create(number=128, velocity=0)

    def test_raises_value_error_negative_velocity(self) -> None:
        with self.assertRaises(ValueError):
            midi_file.PianoNote.create(number=0, velocity=-1)

    def test_raises_value_error_large_velocity(self) -> None:
        with self.assertRaises(ValueError):
            midi_file.PianoNote.create(number=0, velocity=128)


class ConversionMethodsTest(absltest.TestCase):
    def test_note_name_midi_number_consistency(self) -> None:
        name = "C4"
        number = midi_file.note_name_to_midi_number(name)
        self.assertEqual(midi_file.midi_number_to_note_name(number), name)

    def test_key_number_midi_number_consistency(self) -> None:
        key_number = 10
        number = midi_file.key_number_to_midi_number(key_number)
        self.assertEqual(midi_file.midi_number_to_key_number(number), key_number)

    def test_key_number_note_name_consistency(self) -> None:
        key_number = 39
        name = midi_file.key_number_to_note_name(key_number)
        self.assertEqual(midi_file.note_name_to_key_number(name), key_number)


def _get_test_midi(dt: float = 0.01) -> midi_file.MidiFile:
    """A sequence constructed specifically to test hitting a note 2x in a row."""
    seq = music_pb2.NoteSequence()

    # Silence for the first dt.

    # Hit C6 2 times in a row. First one for 1 dt, second one for 3 dt.
    seq.notes.add(
        start_time=1 * dt,
        end_time=2 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=-1,
    )
    seq.notes.add(
        start_time=2.0 * dt,
        end_time=5 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=-1,
    )

    seq.total_time = 5.0 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_test_midi_with_sustain(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # Hit C6 for 1 dt.
    seq.notes.add(
        start_time=0 * dt,
        end_time=1 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=-1,
    )

    # Sustain it for 3 dt.
    seq.control_changes.add(
        time=0 * dt,
        control_number=64,
        control_value=64,
        instrument=0,
    )
    seq.control_changes.add(
        time=3 * dt,
        control_number=64,
        control_value=0,
        instrument=0,
    )

    seq.notes.add(
        start_time=5 * dt,
        end_time=6 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=-1,
    )

    seq.total_time = 6.0 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


class NoteTrajectoryTest(absltest.TestCase):
    def test_same_not_pressed_consecutively(self) -> None:
        midi = _get_test_midi()

        note_traj = midi_file.NoteTrajectory.from_midi(midi, dt=0.01)
        self.assertEqual(len(note_traj), 6)

        self.assertEqual(note_traj.notes[0], [])  # Silence.

        self.assertLen(note_traj.notes[1], 1)
        self.assertEqual(note_traj.notes[1][0].name, "C6")

        # To prevent the note from being sustained, the third timestep should be empty
        # even though the note is played anew at that timestep.
        self.assertEqual(note_traj.notes[2], [])

        # Now the note should be active for 2 timesteps.
        self.assertLen(note_traj.notes[3], 1)
        self.assertEqual(note_traj.notes[3][0].name, "C6")
        self.assertLen(note_traj.notes[4], 1)
        self.assertEqual(note_traj.notes[4][0].name, "C6")

    def test_sustain(self) -> None:
        midi = _get_test_midi_with_sustain()

        note_traj = midi_file.NoteTrajectory.from_midi(midi, dt=0.01)
        self.assertEqual(len(note_traj), 7)

        sustain = note_traj.sustains

        # Sustain should be active for the first 3 timesteps.
        for i in range(3):
            self.assertTrue(sustain[i])

        # Sustain should be inactive for the last 3 timesteps.
        for i in range(3, 6):
            self.assertFalse(sustain[i])


if __name__ == "__main__":
    absltest.main()

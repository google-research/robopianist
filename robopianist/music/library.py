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

"""A library of MIDI songs with fingering information."""

from pathlib import Path
from typing import Callable, Dict

from note_seq.protobuf import music_pb2

from robopianist.music import midi_file

_HERE = Path(__file__).parent
_DATA_PATH = _HERE / "data"


def toy(right_finger: int = 1, left_finger: int = 6) -> midi_file.MidiFile:
    """Toy sequence for testing purposes."""
    seq = music_pb2.NoteSequence()

    # Right hand.
    seq.notes.add(
        start_time=0.0,
        end_time=0.5,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=right_finger,
    )
    seq.notes.add(
        start_time=0.5,
        end_time=1.0,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("G5"),
        part=right_finger,
    )

    # Left hand.
    seq.notes.add(
        start_time=0.0,
        end_time=0.5,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C3"),
        part=left_finger,
    )
    seq.notes.add(
        start_time=0.5,
        end_time=1.0,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C4"),
        part=left_finger,
    )

    seq.total_time = 1.0
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def twinkle_twinkle_little_star_one_hand() -> midi_file.MidiFile:
    """Simple "Twinkle Twinkle Little Star" for the right hand."""

    seq = music_pb2.NoteSequence()

    # Add metadata.
    seq.sequence_metadata.title = "Twinkle Twinkle (one hand)"
    seq.sequence_metadata.artist = "robopianist"

    seq.notes.add(pitch=60, start_time=0.0, end_time=0.5, velocity=80, part=0)  # twin
    seq.notes.add(pitch=60, start_time=0.5, end_time=1.0, velocity=80, part=0)  # kle
    seq.notes.add(pitch=67, start_time=1.0, end_time=1.5, velocity=80, part=2)  # twin
    seq.notes.add(pitch=67, start_time=1.5, end_time=2.0, velocity=80, part=2)  # kle
    seq.notes.add(pitch=69, start_time=2.0, end_time=2.5, velocity=80, part=3)  # lit
    seq.notes.add(pitch=69, start_time=2.5, end_time=3.0, velocity=80, part=3)  # tle
    seq.notes.add(pitch=67, start_time=3.0, end_time=4.0, velocity=80, part=2)  # star

    seq.notes.add(pitch=65, start_time=4.0, end_time=4.5, velocity=80, part=3)  # how
    seq.notes.add(pitch=65, start_time=4.5, end_time=5.0, velocity=80, part=3)  # I
    seq.notes.add(pitch=64, start_time=5.0, end_time=5.5, velocity=80, part=2)  # won
    seq.notes.add(pitch=64, start_time=5.5, end_time=6.0, velocity=80, part=2)  # der
    seq.notes.add(pitch=62, start_time=6.0, end_time=6.5, velocity=80, part=1)  # what
    seq.notes.add(pitch=62, start_time=6.5, end_time=7.0, velocity=80, part=1)  # you
    seq.notes.add(pitch=60, start_time=7.0, end_time=8.0, velocity=80, part=0)  # are

    seq.total_time = 8.0
    seq.tempos.add(qpm=60)

    return midi_file.MidiFile(seq=seq)


def c_major_scale_one_hand(
    right_octave: int = 6,
    note_duration: float = 0.5,
) -> midi_file.MidiFile:
    """C Major scale for the right hand."""
    seq = music_pb2.NoteSequence()

    # Add metadata.
    seq.sequence_metadata.title = "C major scale (one hand)"
    seq.sequence_metadata.artist = "robopianist"

    pitches = [0, 2, 4, 5, 7, 9, 11, 12]

    # Forward.
    rh_fingering = [0, 1, 2, 0, 1, 2, 3, 4]
    for i in range(8):
        seq.notes.add(
            pitch=12 * right_octave + pitches[i],
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )

    # Backward.
    rh_fingering = [3, 2, 1, 0, 2, 1, 0]
    for i in range(7):
        seq.notes.add(
            pitch=12 * right_octave + pitches[7 - i - 1],
            start_time=(8 + i) * note_duration,
            end_time=(9 + i) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )

    seq.total_time = 15 * note_duration
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def d_major_scale_one_hand(
    right_octave: int = 6,
    note_duration: float = 0.5,
) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # Add metadata.
    seq.sequence_metadata.title = "D major scale (one hand)"
    seq.sequence_metadata.artist = "robopianist"

    pitches = [2, 4, 6, 7, 9, 11, 13, 14]

    # Forward.
    rh_fingering = [0, 1, 2, 0, 1, 2, 3, 4]
    for i in range(8):
        seq.notes.add(
            pitch=12 * right_octave + pitches[i],
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )

    # Backward.
    rh_fingering = [3, 2, 1, 0, 2, 1, 0]
    for i in range(7):
        seq.notes.add(
            pitch=12 * right_octave + pitches[7 - i - 1],
            start_time=(8 + i) * note_duration,
            end_time=(9 + i) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )

    seq.total_time = 15 * note_duration
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def c_major_scale_two_hands(
    left_octave: int = 4,
    right_octave: int = 6,
    note_duration: float = 0.5,
) -> midi_file.MidiFile:
    """C Major scale for both hands."""
    seq = music_pb2.NoteSequence()

    # Add metadata.
    seq.sequence_metadata.title = "C major scale"
    seq.sequence_metadata.artist = "robopianist"

    pitches = [0, 2, 4, 5, 7, 9, 11, 12]

    # Forward.
    rh_fingering = [0, 1, 2, 0, 1, 2, 3, 4]
    lh_fingering = [9, 8, 7, 6, 5, 7, 6, 5]
    for i in range(8):
        # Right hand.
        seq.notes.add(
            pitch=12 * right_octave + pitches[i],
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )
        # Left hand.
        seq.notes.add(
            pitch=12 * left_octave + pitches[i],
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=lh_fingering[i],
        )

    # Backward.
    rh_fingering = [3, 2, 1, 0, 2, 1, 0]
    lh_fingering = [6, 7, 5, 6, 7, 8, 9]
    for i in range(7):
        # Right hand.
        seq.notes.add(
            pitch=12 * right_octave + pitches[7 - i - 1],
            start_time=(8 + i) * note_duration,
            end_time=(9 + i) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )
        # Left hand.
        seq.notes.add(
            pitch=12 * left_octave + pitches[7 - i - 1],
            start_time=(8 + i) * note_duration,
            end_time=(9 + i) * note_duration,
            velocity=80,
            part=lh_fingering[i],
        )

    seq.total_time = 15 * note_duration
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def d_major_scale_two_hands(
    left_octave: int = 4,
    right_octave: int = 6,
    note_duration: float = 0.5,
) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # Add metadata.
    seq.sequence_metadata.title = "D major scale"
    seq.sequence_metadata.artist = "robopianist"

    pitches = [2, 4, 6, 7, 9, 11, 13, 14]

    # Forward.
    rh_fingering = [0, 1, 2, 0, 1, 2, 3, 4]
    lh_fingering = [9, 8, 7, 6, 5, 7, 6, 5]
    for i in range(8):
        # Right hand.
        seq.notes.add(
            pitch=12 * right_octave + pitches[i],
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )
        # Left hand.
        seq.notes.add(
            pitch=12 * left_octave + pitches[i],
            start_time=i * note_duration,
            end_time=(i + 1) * note_duration,
            velocity=80,
            part=lh_fingering[i],
        )

    # Backward.
    rh_fingering = [3, 2, 1, 0, 2, 1, 0]
    lh_fingering = [6, 7, 5, 6, 7, 8, 9]
    for i in range(7):
        # Right hand.
        seq.notes.add(
            pitch=12 * right_octave + pitches[7 - i - 1],
            start_time=(8 + i) * note_duration,
            end_time=(9 + i) * note_duration,
            velocity=80,
            part=rh_fingering[i],
        )
        # Left hand.
        seq.notes.add(
            pitch=12 * left_octave + pitches[7 - i - 1],
            start_time=(8 + i) * note_duration,
            end_time=(9 + i) * note_duration,
            velocity=80,
            part=lh_fingering[i],
        )

    seq.total_time = 15 * note_duration
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def c_major_chord_progression_two_hands() -> midi_file.MidiFile:
    """C Major chord progression for both hands."""
    seq = music_pb2.NoteSequence()

    # Add metadata.
    seq.sequence_metadata.title = "C major chord progression"
    seq.sequence_metadata.artist = "robopianist"

    # C Major chord.
    seq.notes.add(pitch=48, start_time=0, end_time=1, velocity=80, part=5)  # Left.
    seq.notes.add(pitch=60, start_time=0, end_time=1, velocity=80, part=0)  # Right.
    seq.notes.add(pitch=64, start_time=0, end_time=1, velocity=80, part=2)  # Right.
    seq.notes.add(pitch=67, start_time=0, end_time=1, velocity=80, part=4)  # Right.

    # F Major chord.
    seq.notes.add(pitch=41, start_time=1, end_time=2, velocity=80, part=8)  # Left.
    seq.notes.add(pitch=65, start_time=1, end_time=2, velocity=80, part=0)  # Right.
    seq.notes.add(pitch=69, start_time=1, end_time=2, velocity=80, part=2)  # Right.
    seq.notes.add(pitch=72, start_time=1, end_time=2, velocity=80, part=4)  # Right.

    # G Major chord.
    seq.notes.add(pitch=43, start_time=2, end_time=3, velocity=80, part=7)  # Left.
    seq.notes.add(pitch=67, start_time=2, end_time=3, velocity=80, part=0)  # Right.
    seq.notes.add(pitch=71, start_time=2, end_time=3, velocity=80, part=2)  # Right.
    seq.notes.add(pitch=74, start_time=2, end_time=3, velocity=80, part=4)  # Right.

    # C Major chord.
    seq.notes.add(pitch=48, start_time=3, end_time=4, velocity=80, part=5)  # Left.
    seq.notes.add(pitch=60, start_time=3, end_time=4, velocity=80, part=0)  # Right.
    seq.notes.add(pitch=64, start_time=3, end_time=4, velocity=80, part=2)  # Right.
    seq.notes.add(pitch=67, start_time=3, end_time=4, velocity=80, part=4)  # Right.

    seq.total_time = 4
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def twinkle_twinkle_rousseau() -> midi_file.MidiFile:
    """A chunk of YouTube channel Rousseau's Twinkle Twinkle with annotated fingering.

    See data/README.md for more information.
    """
    midi = midi_file.MidiFile.from_file(
        _DATA_PATH / "rousseau" / "twinkle-twinkle-trimmed.mid"
    )

    # Add metadata.
    midi.seq.sequence_metadata.artist = "Rousseau"
    midi.seq.sequence_metadata.title = "Twinkle Twinkle (YouTube)"

    FINGERING = [
        1,  # C5
        9,  # C3
        5,  # C4
        0,  # C5
        3,  # G5
        6,  # E4
        3,  # G5
        8,  # C4
        4,  # A5
        5,  # F4
        4,  # A5
        8,  # C4
        3,  # G5
        6,  # E4
        4,  # G5
        8,  # C4
        3,  # F5
        5,  # D4
        3,  # F5
        6,  # B3
        5,  # C4
        2,  # E5
        6,  # A3
        2,  # E5
        1,  # D5
        8,  # F3
        2,  # E5
        1,  # D5
        2,  # E5
        1,  # D5
        6,  # G3
        2,  # E5
        0,  # C5
        9,  # C3
    ]

    sorted_notes = sorted(
        midi.seq.notes,
        key=lambda note: note.start_time,
    )
    assert len(FINGERING) == len(sorted_notes)

    for i, note in enumerate(sorted_notes):
        note.part = FINGERING[i]

    return midi


def nocturne_rousseau() -> midi_file.MidiFile:
    """A chunk of YouTube channel Rousseau's Nocturne with annotated fingering.

    See data/README.md for more information.
    """
    midi = midi_file.MidiFile.from_file(
        _DATA_PATH / "rousseau" / "nocturne-trimmed.mid"
    )

    # Add metadata.
    midi.seq.sequence_metadata.artist = "Rousseau"
    midi.seq.sequence_metadata.title = "Nocturne (YouTube)"

    FINGERING = [
        0,
        8,
        4,
        9,
        6,
        8,
        6,
        5,
        9,
        9,
        6,
        2,
        8,
        6,
        5,
        3,
        8,
        2,
        9,
        6,
        8,
        6,
        5,
        7,
        1,
        9,
        6,
        8,
        6,
        5,
        0,
        7,
        4,
        9,
        6,
        8,
        6,
        5,
        0,
        3,
        2,
        1,
        0,
        7,
        4,
        9,
        6,
        8,
        6,
        5,
        0,
        7,
        2,
        9,
        6,
        7,
        6,
        5,
        7,
        2,
        9,
        6,
        7,
        6,
        5,
        2,
        7,
        1,
        9,
        6,
        8,
        6,
        5,
        9,
        4,
        9,
        5,
        6,
        5,
        5,
        0,
        9,
        2,
        9,
        6,
        7,
        6,
        5,
        9,
        1,
        9,
        6,
        7,
        6,
        5,
        8,
        0,
        9,
        6,
        4,
        8,
        6,
        5,
        3,
        8,
        2,
        1,
        9,
        6,
        0,
        3,
        8,
        5,
        0,
        1,
        7,
        2,
    ]

    sorted_notes = sorted(
        midi.seq.notes,
        key=lambda note: (note.start_time, note.pitch),
    )
    assert len(FINGERING) == len(sorted_notes)

    for i, note in enumerate(sorted_notes):
        note.part = FINGERING[i]

    return midi


MIDI_NAME_TO_CALLABLE: Dict[str, Callable[[], midi_file.MidiFile]] = {
    "TwinkleTwinkleLittleStar": twinkle_twinkle_little_star_one_hand,
    "CMajorScaleOneHand": c_major_scale_one_hand,
    "CMajorScaleTwoHands": c_major_scale_two_hands,
    "DMajorScaleOneHand": d_major_scale_one_hand,
    "DMajorScaleTwoHands": d_major_scale_two_hands,
    "CMajorChordProgressionTwoHands": c_major_chord_progression_two_hands,
    "TwinkleTwinkleRousseau": twinkle_twinkle_rousseau,
    "NocturneRousseau": nocturne_rousseau,
}

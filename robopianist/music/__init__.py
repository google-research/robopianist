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

"""Music module."""

from pathlib import Path
from typing import Union

from robopianist import _PROJECT_ROOT
from robopianist.music import library, midi_file


def _camel_case(name: str) -> str:
    new_name = name.replace("'", "")  # Remove apostrophes.
    new_name = new_name.replace("_", " ").title().replace(" ", "")
    # We have a -{number} suffix which originally came from the different fingering
    # annotations per file in the PIG dataset. We remove it here.
    if "-" in new_name:
        new_name = new_name[: new_name.index("-")]
    return new_name


_PIG_DIR = _PROJECT_ROOT / "robopianist" / "music" / "data" / "pig_single_finger"
_PIG_FILES = sorted(_PIG_DIR.glob("*.proto"))
PIG_MIDIS = [_camel_case(Path(f).stem) for f in _PIG_FILES]
_ETUDE_SUBSET = (
    "french_suite_no_1_allemande-1",
    "french_suite_no_5_sarabande-1",
    "piano_sonata_d_845_1st_mov-1",
    "partita_no_2_6-1",
    "waltz_op_64_no_1-1",
    "bagatelle_op_3_no_4-1",
    "kreisleriana_op_16_no_8-1",
    "french_suite_no_5_gavotte-1",
    "piano_sonata_no_23_2nd_mov-1",
    "golliwogg's_cakewalk-1",
    "piano_sonata_no_2_1st_mov-1",
    "piano_sonata_k_279_in_c_major_1st_mov-1",
)
ETUDE_MIDIS = [_camel_case(name) for name in _ETUDE_SUBSET]
_PIG_NAME_TO_FILE = dict(zip(PIG_MIDIS, _PIG_FILES))
DEBUG_MIDIS = list(library.MIDI_NAME_TO_CALLABLE.keys())
ALL = DEBUG_MIDIS + PIG_MIDIS


def load(
    path_or_name: Union[str, Path],
    stretch: float = 1.0,
    shift: int = 0,
) -> midi_file.MidiFile:
    """Make a MidiFile object from a path or name.

    Args:
        path_or_name: Path or name of the midi file.
        stretch: Temporal stretch factor. Values greater than 1.0 slow down a song, and
            values less than 1.0 speed it up.
        shift: Number of semitones to transpose the song by.

    Returns:
        A MidiFile object.

    Raises:
        ValueError if the path extension is not supported or the MIDI file is invalid.
        KeyError if the name is not found in the library.
    """
    path = Path(path_or_name)

    if path.suffix:  # Note strings will have an empty string suffix.
        midi = midi_file.MidiFile.from_file(path)
    else:
        # Debug midis are generated programmatically and thus should not be loaded from
        # file.
        if path.stem in DEBUG_MIDIS:
            midi = library.MIDI_NAME_TO_CALLABLE[path.stem]()
        # PIG midis are stored as proto files and should be loaded from file.
        elif path.stem in PIG_MIDIS:
            midi = midi_file.MidiFile.from_file(_PIG_NAME_TO_FILE[path.stem])
        else:
            raise KeyError(f"Unknown name: {path.stem}. Available names: {ALL}.")

    return midi.stretch(stretch).transpose(shift)


__all__ = [
    "ALL",
    "DEBUG_MIDIS",
    "PIG_MIDIS",
    "ETUDE_MIDIS",
    "load",
]

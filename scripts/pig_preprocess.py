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

"""Converts the PIG dataset to `note_seq` proto files.

The PIG dataset can be downloaded from: https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset

Example usage:
    python scripts/pig_preprocess.py --dataset_dir ~/Downloads/PianoFingeringDataset_v1.2/
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pretty_midi
from absl import app, flags
from note_seq.protobuf import music_pb2

from robopianist.music import midi_file

_HERE = Path(__file__).parent
_DEFAULT_SAVE_DIR = (
    _HERE.parent / "robopianist" / "music" / "data" / "pig_single_finger"
)

_N_MIDIS = 150

_DATASET_DIR = flags.DEFINE_string(
    "dataset_dir", None, "Where the PIG dataset is located."
)
_SAVE_DIR = flags.DEFINE_string(
    "save_dir", str(_DEFAULT_SAVE_DIR), "Where to save the proto files."
)


@dataclass
class Line:
    note_id: int
    onset_time: float
    offset_time: float
    pitch: str
    onset_velocity: int
    offset_velocity: int
    channel: int
    finger: int

    @staticmethod
    def from_line(line: str) -> "Line":
        parts = line.split("\t")

        # Ignore finger substitutions.
        finger = int(parts[7].split("_")[0])
        if finger < 0:
            finger = abs(finger) + 5
        finger -= 1

        return Line(
            note_id=int(parts[0]),
            onset_time=float(parts[1]),
            offset_time=float(parts[2]),
            pitch=parts[3],
            onset_velocity=int(parts[4]),
            offset_velocity=int(parts[5]),
            channel=int(parts[6]),
            finger=finger,
        )


def main(_) -> None:
    dataset_dir = Path(_DATASET_DIR.value)
    assert dataset_dir.exists()

    save_dir = Path(_SAVE_DIR.value)
    save_dir.mkdir(exist_ok=True, parents=True)

    fingering_dir = dataset_dir / "FingeringFiles"
    assert fingering_dir.exists()
    all_files = list(fingering_dir.glob("*.txt"))
    all_files.sort()

    # Only grab the first fingering version of each piece.
    unique = set()
    fingering_files = []
    for path in all_files:
        unique_name = path.stem.split("-")[0]
        if unique_name not in unique:
            unique.add(unique_name)
            fingering_files.append(path)
    assert len(fingering_files) == _N_MIDIS

    df = pd.read_csv(dataset_dir / "List.csv")

    for sheet in fingering_files:
        name = sheet.stem
        number = int(name.split("-")[1][0])
        index = int(name.split("-")[0])
        piece = (
            df.iloc[index - 1]["Piece"]
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace(",", "")
            .lower()
        )

        with open(sheet, "r") as f:
            lines = f.read().splitlines()

        info = []
        for line in lines[1:]:
            info.append(Line.from_line(line))

        seq = music_pb2.NoteSequence()
        for event in info:
            seq.notes.add(
                start_time=event.onset_time,
                end_time=event.offset_time,
                velocity=event.onset_velocity,
                pitch=pretty_midi.note_name_to_number(event.pitch),
                part=event.finger,
            )
        seq.total_time = info[-1].offset_time

        # Add metadata.
        seq.sequence_metadata.title = piece.replace("_", " ").title()

        # Save proto file.
        filename = save_dir / f"{piece}-{number}.proto"
        midi_file.MidiFile(seq).save(filename)


if __name__ == "__main__":
    flags.mark_flag_as_required("dataset_dir")
    app.run(main)

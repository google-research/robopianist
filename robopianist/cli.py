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

"""Robopianist CLI."""

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pretty_midi
from note_seq.protobuf import music_pb2

import robopianist
from robopianist.music import midi_file

_DEFAULT_SAVE_DIR = (
    robopianist._PROJECT_ROOT / "robopianist" / "music" / "data" / "pig_single_finger"
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        action="store_true",
        help="print the version of robopianist.",
    )

    parser.add_argument(
        "--download-soundfonts",
        action="store_true",
        help="download additional soundfonts.",
    )

    parser.add_argument(
        "--check-pig-exists",
        action="store_true",
        help="check that the PIG dataset was properly downloaded and processed.",
    )

    subparsers = parser.add_subparsers(dest="subparser_name", help="sub-command help")

    player_parser = subparsers.add_parser("player")
    player_parser.add_argument("--midi-file", required=True, help="MIDI file to play.")
    player_parser.add_argument(
        "--stretch", default=1.0, help="Stretch the MIDI file by this factor."
    )
    player_parser.add_argument(
        "--shift", default=0, help="Shift the MIDI file by this many semitones."
    )

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Where the PIG dataset is located.",
    )
    preprocess_parser.add_argument(
        "--save-dir",
        required=False,
        default=str(_DEFAULT_SAVE_DIR),
        help="Where to save the processed proto files.",
    )

    args = parser.parse_args()

    if args.version:
        print(f"robopianist {robopianist.__version__}")
        return

    if args.download_soundfonts:
        # Download soundfonts.
        script = robopianist._PROJECT_ROOT / "scripts" / "get_soundfonts.sh"
        subprocess.run(["bash", str(script)], check=True)

        # Copy soundfonts to robopianist directory.
        dst_dir = robopianist._PROJECT_ROOT / "robopianist" / "soundfonts"
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_dir = robopianist._PROJECT_ROOT / "third_party" / "soundfonts"
        for file in src_dir.glob("*.sf2"):
            shutil.copy(file, dst_dir / file.name)

        return

    if args.check_pig_exists:
        from robopianist import music

        if len(music.PIG_MIDIS) != 150:
            raise ValueError("PIG dataset was not properly downloaded and processed.")
        else:
            print("PIG dataset is ready to use!")
        return

    if args.subparser_name == "player":
        from robopianist import music

        music.load(args.midi_file, stretch=args.stretch, shift=args.shift).play()
        return

    if args.subparser_name == "preprocess":
        _preprocess_pig(Path(args.dataset_dir), Path(args.save_dir))
        return


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


def _preprocess_pig(dataset_dir: Path, save_dir: Path) -> None:
    assert dataset_dir.exists()
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
    assert len(fingering_files) == 150

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

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
import requests
from note_seq.protobuf import music_pb2
from termcolor import cprint
from tqdm import tqdm

import robopianist
from robopianist import music
from robopianist.music import midi_file

# Dataset variables.
_DEFAULT_SAVE_DIR = (
    robopianist._PROJECT_ROOT / "robopianist" / "music" / "data" / "pig_single_finger"
)

# Soundfont variables.
_SOUNDFONT_DIR = robopianist._PROJECT_ROOT / "robopianist" / "soundfonts"
_SOUNDFONTS = {
    "TimGM6mb": "https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw",
    "SalamanderGrandPiano": "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz",
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        action="store_true",
        help="print the version of robopianist.",
    )

    parser.add_argument(
        "--check-pig-exists",
        action="store_true",
        help="check that the PIG dataset was properly downloaded and processed",
    )

    subparsers = parser.add_subparsers(dest="subparser_name", help="sub-command help")

    player_parser = subparsers.add_parser("player")
    player_parser.add_argument("--midi-file", required=True, help="MIDI file to play")
    player_parser.add_argument(
        "--stretch",
        default=1.0,
        help="stretch the MIDI file by this factor",
        type=float,
    )
    player_parser.add_argument(
        "--shift",
        default=0,
        help="shift the MIDI file by this many semitones",
        type=int,
    )

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument(
        "--dataset-dir",
        required=True,
        help="where the PIG dataset is located",
    )
    preprocess_parser.add_argument(
        "--save-dir",
        required=False,
        default=str(_DEFAULT_SAVE_DIR),
        help="where to save the processed proto files",
    )

    soundfont_parser = subparsers.add_parser("soundfont")
    soundfont_parser.add_argument(
        "--list",
        action="store_true",
        help="list all available soundfonts",
    )
    soundfont_parser.add_argument(
        "--change-default",
        action="store_true",
        help="change the default soundfont",
    )
    soundfont_parser.add_argument(
        "--download",
        action="store_true",
        help="download additional soundfonts",
    )

    args = parser.parse_args()

    if args.version:
        print(f"robopianist {robopianist.__version__}")

    elif args.check_pig_exists:
        if len(music.PIG_MIDIS) != 150:
            cprint("PIG dataset was not properly downloaded and processed.", "red")
        else:
            cprint("PIG dataset is ready to use!", "green")

    elif args.subparser_name == "player":
        music.load(args.midi_file, stretch=args.stretch, shift=args.shift).play()

    elif args.subparser_name == "preprocess":
        _preprocess_pig(Path(args.dataset_dir), Path(args.save_dir))

    elif args.subparser_name == "soundfont":
        if args.list:
            _list_soundfonts()
        elif args.change_default:
            _change_default_soundfont()
        elif args.download:
            _download_soundfont()


def _list_soundfonts() -> None:
    sf2s = _SOUNDFONT_DIR.glob("*.sf2")
    soundfonts = [sf2.stem for sf2 in sf2s]
    is_default = [sf2 == robopianist.SF2_PATH.stem for sf2 in soundfonts]

    print("Available soundfonts:")
    for i, soundfont in enumerate(soundfonts):
        print(f"  ({i}) {soundfont} {'(default)' if is_default[i] else ''}")


def _set_default_soundfont(name: str) -> None:
    # Create a .robopianistrc file if it doesn't exist.
    rc_file = Path.home() / ".robopianistrc"
    if not rc_file.exists():
        rc_file.touch()

    # Check that the soundfont exists.
    soundfont = _SOUNDFONT_DIR / f"{name}.sf2"
    if not soundfont.exists():
        cprint(f"The soundfont {name} does not exist.", "red")
        return

    # Look for the line DEFAULT_SOUNDFONT={} in the .robopianistrc file.
    with rc_file.open("r") as f:
        lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith("DEFAULT_SOUNDFONT="):
            lines[i] = f"DEFAULT_SOUNDFONT={name}\n"
            found = True
            break
    if not found:
        lines.append(f"DEFAULT_SOUNDFONT={name}\n")
    with rc_file.open("w") as f:
        f.writelines(lines)

    cprint(f"Default soundfont set to {name}.", "green")


def _change_default_soundfont() -> None:
    # Get a list of available soundfonts.
    sf2s = _SOUNDFONT_DIR.glob("*.sf2")
    soundfonts = [sf2.stem for sf2 in sf2s]
    is_default = [sf2 == robopianist.SF2_PATH.stem for sf2 in soundfonts]

    print("Available soundfonts:")
    for i, soundfont in enumerate(soundfonts):
        print(f"  ({i}) {soundfont} {'(default)' if is_default[i] else ''}")

    # Get the user's choice.
    choice = input("Enter the soundfont you want to use: ")
    try:
        number = int(choice)
        if number < 0 or number >= len(soundfonts):
            raise ValueError
    except ValueError:
        cprint("Invalid choice.", "red")
        return

    # Set the default soundfont.
    _set_default_soundfont(soundfonts[number])


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


def _download_file(url: str) -> None:
    chunk_size = 1024
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    pbar = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(url.split("/")[-1], "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def _download_soundfont() -> None:
    soundfont_names = list(_SOUNDFONTS.keys())

    is_downloaded = {}
    for sf2 in soundfont_names:
        if (_SOUNDFONT_DIR / f"{sf2}.sf2").exists():
            is_downloaded[sf2] = True
        else:
            is_downloaded[sf2] = False

    print("Which soundfont would you like to download?")
    for i, soundfont in enumerate(_SOUNDFONTS.keys()):
        print(
            f"  ({i}) {soundfont} ({'downloaded' if is_downloaded[soundfont] else 'not downloaded'})"
        )

    # Get the user's choice.
    choice = input("Enter the soundfont you want to download: ")
    try:
        number = int(choice)
        if number < 0 or number >= len(_SOUNDFONTS):
            raise ValueError
    except ValueError:
        cprint("Invalid choice.", "red")
        return

    # Download the soundfont.
    _download_file(_SOUNDFONTS[soundfont_names[number]])

    # Custom extraction logic for each soundfont.
    if soundfont_names[number] == "TimGM6mb":
        shutil.move("TimGM6mb.sf2?format=raw", _SOUNDFONT_DIR / "TimGM6mb.sf2")
    elif soundfont_names[number] == "SalamanderGrandPiano":
        subprocess.run(
            [
                "tar",
                "-xvf",
                "SalamanderGrandPiano-SF2-V3+20200602.tar.xz",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        shutil.move(
            "SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2",
            _SOUNDFONT_DIR / "SalamanderGrandPiano.sf2",
        )
        subprocess.run(
            [
                "rm",
                "-r",
                "SalamanderGrandPiano-SF2-V3+20200602.tar.xz",
                "SalamanderGrandPiano-SF2-V3+20200602",
            ],
        )

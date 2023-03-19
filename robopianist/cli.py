"""Robopianist CLI."""

import argparse
import shutil
import subprocess

import robopianist


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

    subparsers = parser.add_subparsers(dest="subparser_name", help="sub-command help")

    player_parser = subparsers.add_parser("player")
    player_parser.add_argument("--midi-file", default=None, help="MIDI file to play.")
    player_parser.add_argument(
        "--stretch", default=1.0, help="Stretch the MIDI file by this factor."
    )
    player_parser.add_argument(
        "--shift", default=0, help="Shift the MIDI file by this many semitones."
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

    if args.subparser_name == "player":
        if args.midi_file is not None:
            from robopianist import music

            music.load(args.midi_file, stretch=args.stretch, shift=args.shift).play()
        else:
            player_parser.print_help()

        return

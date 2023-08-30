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

"""Play a MIDI file using FluidSynth and PyAudio.

Example usage:
    python examples/play_midi_file.py --file robopianist/music/data/rousseau/nocturne-trimmed.mid
"""

from absl import app, flags

from robopianist import music

_FILE = flags.DEFINE_string("file", None, "Path to the MIDI file.")
_STRETCH = flags.DEFINE_float("stretch", 1.0, "Stretch the MIDI file by this factor.")
_SHIFT = flags.DEFINE_integer("shift", 0, "Shift the MIDI file by this many semitones.")


def main(_) -> None:
    music.load(
        _FILE.value, stretch=_STRETCH.value, shift=_SHIFT.value
    ).trim_silence().play()


if __name__ == "__main__":
    flags.mark_flag_as_required("file")
    app.run(main)

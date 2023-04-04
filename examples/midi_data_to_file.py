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

"""Save a programmatically generated NoteSequence to a MIDI file.

Example usage:
    python examples/midi_data_to_file.py --name CMajorScaleTwoHands --save_path /tmp/c_major_scale_two_hands.mid
"""

from absl import app, flags

from robopianist import music

_NAME = flags.DEFINE_string("name", None, "")
_SAVE_PATH = flags.DEFINE_string("save_path", None, "")


def main(_) -> None:
    music.load(_NAME.value).save(_SAVE_PATH.value)


if __name__ == "__main__":
    flags.mark_flags_as_required(["name", "save_path"])
    app.run(main)

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

from pathlib import Path

__version__ = "1.0.7"

# Path to the root of the project.
_PROJECT_ROOT = Path(__file__).parent.parent

# Path to the soundfont SF2 file.
_SOUNDFONT_PATH = _PROJECT_ROOT / "robopianist" / "soundfonts"
_DEFAULT_SF2_PATH = _SOUNDFONT_PATH / "TimGM6mb.sf2"
_SALAMANDER_SF2_PATH = _SOUNDFONT_PATH / "SalamanderGrandPiano.sf2"
if _SALAMANDER_SF2_PATH.exists():
    SF2_PATH = _SALAMANDER_SF2_PATH
else:
    if not _DEFAULT_SF2_PATH.exists():
        raise FileNotFoundError(
            f"The default soundfont file {_DEFAULT_SF2_PATH} does not exist. Make "
            "sure you have first run `bash scripts/install_deps.sh` in the root of "
            "the project directory."
        )
    SF2_PATH = _DEFAULT_SF2_PATH


__all__ = [
    "__version__",
    "SF2_PATH",
]

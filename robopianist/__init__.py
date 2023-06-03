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

__version__ = "1.0.9"

# Path to the root of the project.
_PROJECT_ROOT = Path(__file__).parent.parent

# Path to the soundfont directory.
_SOUNDFONT_PATH = _PROJECT_ROOT / "robopianist" / "soundfonts"

# TimGM6mb.sf2 is the default soundfont file that is packaged with the pip install or
# available when installing from source using `bash scripts/install_deps.sh`.
_DEFAULT_SF2_PATH = _SOUNDFONT_PATH / "TimGM6mb.sf2"

# We first check if the user has a .robopianistrc file in their home directory. If so,
# we check if it specifies a soundfont file. If so, we use that.
_RC_FILE = Path.home() / ".robopianistrc"
if _RC_FILE.exists():
    found = False
    with open(_RC_FILE, "r") as f:
        for line in f:
            if line.startswith("DEFAULT_SOUNDFONT="):
                soundfont_path = line.split("=")[1].strip()
                SF2_PATH = _SOUNDFONT_PATH / f"{soundfont_path}.sf2"
                if not SF2_PATH.exists():
                    SF2_PATH = _DEFAULT_SF2_PATH
                found = True
                break
    if not found:
        SF2_PATH = _DEFAULT_SF2_PATH
# Otherwise, we look in the soundfont directory. Our preference is for the higher
# quality SalamanderGrandPiano.sf2, but if that is not found, we fall back to the
# default soundfont file.
else:
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

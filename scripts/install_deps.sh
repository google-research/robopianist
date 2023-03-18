#!/bin/bash
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
#
# Install dependencies (macOS and Linux).

set -x

# Install fluidsynth and portaudio.
if [[ $OSTYPE == darwin* ]]; then
    # Install homebrew if not installed.
    if ! command -v brew; then
        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    else
        brew update
    fi
    brew install portaudio fluid-synth ffmpeg
elif [[ $OSTYPE == linux* ]]; then
    sudo apt update
    sudo apt install -y build-essential wget
    sudo apt install -y fluidsynth portaudio19-dev ffmpeg
else
    echo "Unsupported OS"
fi

# Download TimGM6mb.sf2 soundfont.
mkdir -p third_party/soundfonts
LINK=https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw
wget $LINK -O third_party/soundfonts/TimGM6mb.sf2

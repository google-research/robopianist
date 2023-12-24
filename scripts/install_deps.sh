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
# Command line arguments:
#   --no-soundfonts: Skip downloading soundfonts.

set -x

SKIP_SOUNDFONTS=false
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --no-soundfonts)
            SKIP_SOUNDFONTS=true
            shift
            ;;
        *)
            echo "Unknown argument: $key"
            exit 1
            ;;
    esac
done

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

# Install soundfonts.
if [ "$SKIP_SOUNDFONTS" = false ]; then
    # Download TimGM6mb.sf2 soundfont.
    mkdir -p third_party/soundfonts
    LINK=https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw
    if [ ! -f third_party/soundfonts/TimGM6mb.sf2 ]; then
        wget $LINK -O third_party/soundfonts/TimGM6mb.sf2
    fi

    # Copy soundfonts to robopianist.
    mkdir -p robopianist/soundfonts
    if [ ! -d "third_party/soundfonts" ]; then
        echo "third_party/soundfonts does not exist. Run scripts/get_soundfonts.sh first."
        exit 1
    fi
    cp -r third_party/soundfonts/* robopianist/soundfonts
fi

# Copy shadow_hand menagerie model to robopianist.
cd third_party/mujoco_menagerie
git checkout 1afc8be64233dcfe943b2fe0c505ec1e87a0a13e
cd ../..
mkdir -p robopianist/models/hands/third_party/shadow_hand
if [ ! -d "third_party/mujoco_menagerie/shadow_hand" ]; then
    echo "third_party/mujoco_menagerie/shadow_hand does not exist. Run git submodule init && git submodule update first."
    exit 1
fi
cp -r third_party/mujoco_menagerie/shadow_hand/* robopianist/models/hands/third_party/shadow_hand

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
# Install additional soundfonts.

set -ex

mkdir -p third_party/soundfonts

# Salamander Grand Piano.
LINK=https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz
wget $LINK
tar -xvf SalamanderGrandPiano-SF2-V3+20200602.tar.xz
mv SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2 third_party/soundfonts/SalamanderGrandPiano.sf2
rm -r SalamanderGrandPiano-SF2-V3+20200602.tar.xz SalamanderGrandPiano-SF2-V3+20200602

# Fluid R3 (GM).
LINK=https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip
wget $LINK
unzip -j FluidR3_GM.zip FluidR3_GM.sf2
mv FluidR3_GM.sf2 third_party/soundfonts/
rm -rf FluidR3_GM.zip

# MuseScore_General.sf2.
LINK=https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2
wget $LINK -O third_party/soundfonts/MuseScore_General.sf2

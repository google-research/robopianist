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

from robopianist.suite.tasks.base import PianoOnlyTask, PianoTask
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from robopianist.suite.tasks.self_actuated_piano import SelfActuatedPiano

__all__ = [
    "PianoTask",
    "PianoOnlyTask",
    "SelfActuatedPiano",
    "PianoWithShadowHands",
    "PianoWithOneShadowHand",
]

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

from typing import Dict, Tuple

from robopianist import MENAGERIE_ROOT

NQ = 24  # Number of joints.
NU = 20  # Number of actuators.

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("WRJ1", "WRJ0"),
    "thumb": ("THJ4", "THJ3", "THJ2", "THJ1", "THJ0"),
    "first": ("FFJ3", "FFJ2", "FFJ1", "FFJ0"),
    "middle": ("MFJ3", "MFJ2", "MFJ1", "MFJ0"),
    "ring": ("RFJ3", "RFJ2", "RFJ1", "RFJ0"),
    "little": ("LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0"),
}

FINGERTIP_BODIES: Tuple[str, ...] = (
    # Important: the order of these names should not be changed.
    "thdistal",
    "ffdistal",
    "mfdistal",
    "rfdistal",
    "lfdistal",
)

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    # Important: the order of these colors should not be changed.
    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.8, 0.8),  # Cyan.
    (0.2, 0.2, 0.8),  # Blue.
    (0.8, 0.8, 0.2),  # Yellow.
)

# Path to the shadow hand E3M5 XML file.
RIGHT_SHADOW_HAND_XML = MENAGERIE_ROOT / "shadow_hand" / "right_hand.xml"
LEFT_SHADOW_HAND_XML = MENAGERIE_ROOT / "shadow_hand" / "left_hand.xml"

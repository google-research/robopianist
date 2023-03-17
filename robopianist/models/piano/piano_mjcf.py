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

"""Programatically build a piano MJCF model."""

import math

from dm_control import mjcf
from mujoco_utils import types

from robopianist.models.piano import piano_constants as consts


def build(add_actuators: bool = False) -> types.MjcfRootElement:
    """Programatically build a piano MJCF.

    Args:
        add_actuators: Whether to add actuators to the piano keys.
    """
    root = mjcf.RootElement()
    root.model = "piano"

    root.compiler.autolimits = True
    root.compiler.angle = "radian"

    # Add materials.
    root.asset.add("material", name="white", rgba=consts.WHITE_KEY_COLOR)
    root.asset.add("material", name="black", rgba=consts.BLACK_KEY_COLOR)

    root.default.geom.type = "box"
    root.default.joint.type = "hinge"
    root.default.joint.axis = [0, 1, 0]
    root.default.site.type = "box"
    root.default.site.group = 4
    root.default.site.rgba = [1, 0, 0, 1]

    # This effectively disables key-key collisions but still allows hand-key collisions,
    # assuming we've kept the default hand contype = conaffinity = 1.
    # See https://mujoco.readthedocs.io/en/latest/computation.html#selection for more
    # details.
    root.default.geom.contype = 0
    root.default.geom.conaffinity = 1

    # Actuator defaults (torque control).
    if add_actuators:
        root.default.general.dyntype = "none"
        root.default.general.dynprm = (consts.ACTUATOR_DYNPRM, 0, 0)
        root.default.general.gaintype = "fixed"
        root.default.general.gainprm = (consts.ACTUATOR_GAINPRM, 0, 0)
        root.default.general.biastype = "none"
        root.default.general.biasprm = (0, 0, 0)

    # White key defaults.
    white_default = root.default.add("default", dclass="white_key")
    white_default.geom.material = "white"
    white_default.geom.size = [
        consts.WHITE_KEY_LENGTH / 2,
        consts.WHITE_KEY_WIDTH / 2,
        consts.WHITE_KEY_HEIGHT / 2,
    ]
    white_default.geom.mass = consts.WHITE_KEY_MASS
    white_default.site.size = white_default.geom.size
    white_default.joint.pos = [-consts.WHITE_KEY_LENGTH / 2, 0, 0]
    white_default.joint.damping = consts.WHITE_JOINT_DAMPING
    white_default.joint.armature = consts.WHITE_JOINT_ARMATURE
    white_default.joint.stiffness = consts.WHITE_KEY_STIFFNESS
    white_default.joint.springref = consts.WHITE_KEY_SPRINGREF * math.pi / 180
    white_default.joint.range = [0, consts.WHITE_KEY_JOINT_MAX_ANGLE]
    if add_actuators:
        white_default.general.ctrlrange = [0, consts.WHITE_KEY_JOINT_MAX_ANGLE]

    # Black key defaults.
    black_default = root.default.add("default", dclass="black_key")
    black_default.geom.material = "black"
    black_default.geom.size = [
        consts.BLACK_KEY_LENGTH / 2,
        consts.BLACK_KEY_WIDTH / 2,
        consts.BLACK_KEY_HEIGHT / 2,
    ]
    black_default.site.size = black_default.geom.size
    black_default.geom.mass = consts.BLACK_KEY_MASS
    black_default.joint.pos = [-consts.BLACK_KEY_LENGTH / 2, 0, 0]
    black_default.joint.damping = consts.BLACK_JOINT_DAMPING
    black_default.joint.armature = consts.BLACK_JOINT_ARMATURE
    black_default.joint.stiffness = consts.BLACK_KEY_STIFFNESS
    black_default.joint.springref = consts.BLACK_KEY_SPRINGREF * math.pi / 180
    black_default.joint.range = [0, consts.BLACK_KEY_JOINT_MAX_ANGLE]
    if add_actuators:
        black_default.general.ctrlrange = [0, consts.BLACK_KEY_JOINT_MAX_ANGLE]

    # Add base.
    base_body = root.worldbody.add("body", name="base", pos=consts.BASE_POS)
    base_body.add("geom", type="box", size=consts.BASE_SIZE, rgba=consts.BASE_COLOR)

    WHITE_KEY_INDICES = [
        0,
        2,
        3,
        5,
        7,
        8,
        10,
        12,
        14,
        15,
        17,
        19,
        20,
        22,
        24,
        26,
        27,
        29,
        31,
        32,
        34,
        36,
        38,
        39,
        41,
        43,
        44,
        46,
        48,
        50,
        51,
        53,
        55,
        56,
        58,
        60,
        62,
        63,
        65,
        67,
        68,
        70,
        72,
        74,
        75,
        77,
        79,
        80,
        82,
        84,
        86,
        87,
    ]

    # These will hold kwargs. We'll subsequently use them to create the actual objects.
    geoms = []
    bodies = []
    joints = []
    sites = []
    actuators = []

    for i in range(consts.NUM_WHITE_KEYS):
        y_coord = (
            -consts.PIANO_LENGTH * 0.5
            + consts.WHITE_KEY_WIDTH * 0.5
            + i * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
        )
        bodies.append(
            {
                "name": f"white_key_{WHITE_KEY_INDICES[i]}",
                "pos": [consts.WHITE_KEY_X_OFFSET, y_coord, consts.WHITE_KEY_Z_OFFSET],
            }
        )
        geoms.append(
            {
                "name": f"white_key_geom_{WHITE_KEY_INDICES[i]}",
                "dclass": "white_key",
            }
        )
        joints.append(
            {
                "name": f"white_joint_{WHITE_KEY_INDICES[i]}",
                "dclass": "white_key",
            }
        )
        sites.append(
            {
                "name": f"white_key_site_{WHITE_KEY_INDICES[i]}",
                "dclass": "white_key",
            }
        )
        if add_actuators:
            actuators.append(
                {
                    "joint": f"white_joint_{WHITE_KEY_INDICES[i]}",
                    "name": f"white_actuator_{WHITE_KEY_INDICES[i]}",
                    "dclass": "white_key",
                }
            )

    BLACK_TWIN_KEY_INDICES = [
        4,
        6,
        16,
        18,
        28,
        30,
        40,
        42,
        52,
        54,
        64,
        66,
        76,
        78,
    ]
    BLACK_TRIPLET_KEY_INDICES = [
        1,
        9,
        11,
        13,
        21,
        23,
        25,
        33,
        35,
        37,
        45,
        47,
        49,
        57,
        59,
        61,
        69,
        71,
        73,
        81,
        83,
        85,
    ]

    # Place the lone black key on the far left.
    y_coord = consts.WHITE_KEY_WIDTH + 0.5 * (
        -consts.PIANO_LENGTH + consts.SPACING_BETWEEN_WHITE_KEYS
    )
    bodies.append(
        {
            "name": f"black_key_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "pos": [consts.BLACK_KEY_X_OFFSET, y_coord, consts.BLACK_KEY_Z_OFFSET],
        }
    )
    geoms.append(
        {
            "name": f"black_key_geom_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "dclass": "black_key",
        }
    )
    joints.append(
        {
            "name": f"black_joint_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "dclass": "black_key",
        }
    )
    sites.append(
        {
            "name": f"black_key_site_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "dclass": "black_key",
        }
    )
    if add_actuators:
        actuators.append(
            {
                "joint": f"black_joint_{BLACK_TRIPLET_KEY_INDICES[0]}",
                "name": f"black_actuator_{BLACK_TRIPLET_KEY_INDICES[0]}",
                "dclass": "black_key",
            }
        )

    # Place the twin black keys.
    n = 0
    TWIN_INDICES = list(range(2, consts.NUM_WHITE_KEYS - 1, 7))
    for twin_index in TWIN_INDICES:
        for j in range(2):
            y_coord = (
                -consts.PIANO_LENGTH * 0.5
                + (j + 1) * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
                + twin_index
                * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
            )
            bodies.append(
                {
                    "name": f"black_key_{BLACK_TWIN_KEY_INDICES[n]}",
                    "pos": [
                        consts.BLACK_KEY_X_OFFSET,
                        y_coord,
                        consts.BLACK_KEY_Z_OFFSET,
                    ],
                }
            )
            geoms.append(
                {
                    "name": f"black_key_geom_{BLACK_TWIN_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            joints.append(
                {
                    "name": f"black_joint_{BLACK_TWIN_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            sites.append(
                {
                    "name": f"black_key_site_{BLACK_TWIN_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            if add_actuators:
                actuators.append(
                    {
                        "joint": f"black_joint_{BLACK_TWIN_KEY_INDICES[n]}",
                        "name": f"black_actuator_{BLACK_TWIN_KEY_INDICES[n]}",
                        "dclass": "black_key",
                    }
                )
            n += 1

    # Place the triplet black keys.
    n = 1  # Skip the lone black key.
    TRIPLET_INDICES = list(range(5, consts.NUM_WHITE_KEYS - 1, 7))
    for triplet_index in TRIPLET_INDICES:
        for j in range(3):
            y_coord = (
                -consts.PIANO_LENGTH * 0.5
                + (j + 1) * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
                + triplet_index
                * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
            )
            bodies.append(
                {
                    "name": f"black_key_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "pos": [
                        consts.BLACK_KEY_X_OFFSET,
                        y_coord,
                        consts.BLACK_KEY_Z_OFFSET,
                    ],
                }
            )
            geoms.append(
                {
                    "name": f"black_key_geom_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            joints.append(
                {
                    "name": f"black_joint_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            sites.append(
                {
                    "name": f"black_key_site_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            if add_actuators:
                actuators.append(
                    {
                        "joint": f"black_joint_{BLACK_TRIPLET_KEY_INDICES[n]}",
                        "name": f"black_actuator_{BLACK_TRIPLET_KEY_INDICES[n]}",
                        "dclass": "black_key",
                    }
                )
            n += 1

    # Sort the elements based on the key number.
    names: list[str] = [body["name"] for body in bodies]  # type: ignore
    indices = sorted(range(len(names)), key=lambda k: int(names[k].split("_")[-1]))
    bodies = [bodies[i] for i in indices]
    geoms = [geoms[i] for i in indices]
    joints = [joints[i] for i in indices]
    sites = [sites[i] for i in indices]
    if add_actuators:
        actuators = [actuators[i] for i in indices]

    # Now create the corresponding MJCF elements and add them to the root.
    for i in range(len(bodies)):
        body = root.worldbody.add("body", **bodies[i])
        body.add("geom", **geoms[i])
        body.add("joint", **joints[i])
        body.add("site", **sites[i])
        if add_actuators:
            root.actuator.add("general", **actuators[i])

    return root

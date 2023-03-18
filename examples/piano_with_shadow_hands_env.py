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

"""Piano with shadow hands environment."""

import dm_env
import numpy as np
from absl import app, flags
from dm_control.mjcf import export_with_assets
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco import viewer as mujoco_viewer
from mujoco_utils import composer_utils

from robopianist import music, viewer
from robopianist.suite.tasks import piano_with_shadow_hands
from robopianist.wrappers import PianoSoundVideoWrapper

_FILE = flags.DEFINE_string("file", "TwinkleTwinkleRousseau", "")
_CONTROL_TIMESTEP = flags.DEFINE_float("control_timestep", 0.05, "")
_STRETCH = flags.DEFINE_float("stretch", 1.0, "")
_SHIFT = flags.DEFINE_integer("shift", 0, "")
_RECORD = flags.DEFINE_bool("record", False, "")
_EXPORT = flags.DEFINE_bool("export", False, "")
_GRAVITY_COMPENSATION = flags.DEFINE_bool("gravity_compensation", False, "")
_HEADLESS = flags.DEFINE_bool("headless", False, "")
_TRIM_SILENCE = flags.DEFINE_bool("trim_silence", False, "")
_PRIMITIVE_FINGERTIP_COLLISIONS = flags.DEFINE_bool(
    "primitive_fingertip_collisions", False, ""
)
_REDUCED_ACTION_SPACE = flags.DEFINE_bool("reduced_action_space", False, "")
_DISABLE_FINGERING_REWARD = flags.DEFINE_bool("disable_fingering_reward", False, "")
_DISABLE_FOREARM_REWARD = flags.DEFINE_bool("disable_forearm_reward", False, "")
_DISABLE_COLORIZATION = flags.DEFINE_bool("disable_colorization", False, "")
_DISABLE_HAND_COLLISIONS = flags.DEFINE_bool("disable_hand_collisions", False, "")
_CANONICALIZE = flags.DEFINE_bool("canonicalize", False, "")
_N_STEPS_LOOKAHEAD = flags.DEFINE_integer("n_steps_lookahead", 1, "")
_ATTACHMENT_YAW = flags.DEFINE_float("attachment_yaw", 0.0, "")
_ACTION_SEQUENCE = flags.DEFINE_string(
    "action_sequence",
    None,
    "Path to an npy file containing a sequence of actions to replay.",
)


def main(_) -> None:
    task = piano_with_shadow_hands.PianoWithShadowHands(
        change_color_on_activation=True,
        midi=music.load(_FILE.value, stretch=_STRETCH.value, shift=_SHIFT.value),
        trim_silence=_TRIM_SILENCE.value,
        control_timestep=_CONTROL_TIMESTEP.value,
        gravity_compensation=_GRAVITY_COMPENSATION.value,
        primitive_fingertip_collisions=_PRIMITIVE_FINGERTIP_COLLISIONS.value,
        reduced_action_space=_REDUCED_ACTION_SPACE.value,
        n_steps_lookahead=_N_STEPS_LOOKAHEAD.value,
        disable_fingering_reward=_DISABLE_FINGERING_REWARD.value,
        disable_forearm_reward=_DISABLE_FOREARM_REWARD.value,
        disable_colorization=_DISABLE_COLORIZATION.value,
        disable_hand_collisions=_DISABLE_HAND_COLLISIONS.value,
        attachment_yaw=_ATTACHMENT_YAW.value,
    )
    if _EXPORT.value:
        export_with_assets(
            task.root_entity.mjcf_model,
            out_dir="/tmp/robopianist/piano_with_shadow_hands",
            out_file_name="scene.xml",
        )
        mujoco_viewer.launch_from_path(
            "/tmp/robopianist/piano_with_shadow_hands/scene.xml"
        )
        return

    env = composer_utils.Environment(
        task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
    )
    if _RECORD.value:
        env = PianoSoundVideoWrapper(env, record_every=1)
    if _CANONICALIZE.value:
        env = CanonicalSpecWrapper(env)

    action_spec = env.action_spec()
    zeros = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    zeros[-1] = -1.0  # Disable sustain pedal.
    print(f"Action dimension: {action_spec.shape}")

    # Sanity check observables.
    timestep = env.reset()
    dim = 0
    for k, v in timestep.observation.items():
        print(f"\t{k}: {v.shape} {v.dtype}")
        dim += int(np.prod(v.shape))
    print(f"Observation dimension: {dim}")

    print(f"Control frequency: {1 / _CONTROL_TIMESTEP.value} Hz")

    class Policy:
        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            if _ACTION_SEQUENCE.value is not None:
                self._idx = 0
                self._actions = np.load(_ACTION_SEQUENCE.value)

        def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
            del timestep  # Unused.
            if _ACTION_SEQUENCE.value is not None:
                actions = self._actions[self._idx]
                self._idx += 1
                return actions
            return zeros

    policy = Policy()

    if not _RECORD.value:
        if _HEADLESS.value:
            timestep = env.reset()
            while not timestep.last():
                action = policy(timestep)
                timestep = env.step(action)
        else:
            viewer.launch(env, policy=policy)
    else:
        timestep = env.reset()
        while not timestep.last():
            action = policy(timestep)
            timestep = env.step(action)


if __name__ == "__main__":
    app.run(main)

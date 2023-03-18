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

"""Self-actuated piano environment."""

import dm_env
import numpy as np
from absl import app, flags
from dm_control.mjcf import export_with_assets
from mujoco import viewer as mujoco_viewer
from mujoco_utils import composer_utils

from robopianist import music, viewer
from robopianist.suite.tasks import self_actuated_piano
from robopianist.wrappers import PianoSoundVideoWrapper

_FILE = flags.DEFINE_string("file", "TwinkleTwinkleRousseau", "")
_RECORD = flags.DEFINE_bool("record", False, "")
_EXPORT = flags.DEFINE_bool("export", False, "")
_TRIM_SILENCE = flags.DEFINE_bool("trim_silence", False, "")
_CONTROL_TIMESTEP = flags.DEFINE_float("control_timestep", 0.01, "")
_STRETCH = flags.DEFINE_float("stretch", 1.0, "")
_SHIFT = flags.DEFINE_integer("shift", 0, "")
_PLAYBACK_SPEED = flags.DEFINE_float("playback_speed", 1.0, "")


def main(_) -> None:
    task = self_actuated_piano.SelfActuatedPiano(
        midi=music.load(_FILE.value, stretch=_STRETCH.value, shift=_SHIFT.value),
        change_color_on_activation=True,
        trim_silence=_TRIM_SILENCE.value,
        control_timestep=_CONTROL_TIMESTEP.value,
    )
    if _EXPORT.value:
        export_with_assets(
            task.root_entity.mjcf_model,
            out_dir="/tmp/robopianist/self_actuated_piano",
            out_file_name="scene.xml",
        )
        mujoco_viewer.launch_from_path("/tmp/robopianist/self_actuated_piano/scene.xml")
        return

    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )
    if _RECORD.value:
        env = PianoSoundVideoWrapper(
            env,
            record_every=1,
            camera_id="piano/topdown",
            playback_speed=_PLAYBACK_SPEED.value,
        )

    action_spec = env.action_spec()
    min_ctrl = action_spec.minimum
    max_ctrl = action_spec.maximum
    print(f"Action dimension: {action_spec.shape}")

    # Sanity check observables.
    print("Observables:")
    timestep = env.reset()
    dim = 0
    for k, v in timestep.observation.items():
        print(f"\t{k}: {v.shape} {v.dtype}")
        dim += np.prod(v.shape)
    print(f"Observation dimension: {dim}")

    print(f"Control frequency: {1 / _CONTROL_TIMESTEP.value} Hz")

    class Oracle:
        def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
            if timestep.reward is not None:
                assert timestep.reward == 0
            # Only grab the next timestep's goal state.
            goal = timestep.observation["goal"][: task.piano.n_keys]
            key_idxs = np.flatnonzero(goal)
            # For goal keys that should be pressed, set the action to the maximum
            # actuator value. For goal keys that should be released, set the action to
            # the minimum actuator value.
            action = min_ctrl.copy()
            action[key_idxs] = max_ctrl[key_idxs]
            # Grab the sustain pedal action.
            action[-1] = timestep.observation["goal"][-1]
            return action

    policy = Oracle()

    if not _RECORD.value:
        viewer.launch(env, policy=policy)
    else:
        timestep = env.reset()
        while not timestep.last():
            action = policy(timestep)
            timestep = env.step(action)


if __name__ == "__main__":
    app.run(main)

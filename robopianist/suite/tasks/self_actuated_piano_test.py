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

"""Tests for self_actuated_piano.py."""

import numpy as np
from absl.testing import absltest, parameterized
from dm_control import composer
from note_seq.protobuf import music_pb2

from robopianist.music import midi_file
from robopianist.suite.tasks import self_actuated_piano


def _get_test_midi(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # C6 for 2 dts.
    seq.notes.add(
        start_time=0.0,
        end_time=2 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=-1,
    )
    # G5 for 1 dt.
    seq.notes.add(
        start_time=2 * dt,
        end_time=3 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("G5"),
        part=-1,
    )

    seq.total_time = 3 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_env(
    control_timestep: float = 0.01,
    n_steps_lookahead: int = 0,
    reward_type: self_actuated_piano.RewardType = self_actuated_piano.RewardType.NEGATIVE_L2,
) -> composer.Environment:
    task = self_actuated_piano.SelfActuatedPiano(
        midi=_get_test_midi(dt=control_timestep),
        n_steps_lookahead=n_steps_lookahead,
        reward_type=reward_type,
        control_timestep=control_timestep,
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class SelfActuatedPianoTest(parameterized.TestCase):
    def test_observables(self) -> None:
        env = _get_env()
        timestep = env.reset()

        self.assertIn("piano/activation", timestep.observation)
        self.assertIn("piano/sustain_activation", timestep.observation)
        self.assertIn("goal", timestep.observation)

    def test_action_spec(self) -> None:
        env = _get_env()
        self.assertEqual(env.action_spec().shape, (env.task.piano.n_keys + 1,))

    def test_termination_and_discount(self) -> None:
        env = _get_env()
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        # With a dt of 0.01 and a 3 dt long midi, the episode should end after 4 steps.
        for _ in range(3):
            timestep = env.step(zero_action)
            self.assertFalse(env.task.should_terminate_episode(env.physics))
            np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

        # 1 more step to terminate.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # No failure, so discount should be 1.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

    @parameterized.parameters(0, 1, 2, 5)
    def test_goal_observable_lookahead(self, n_steps_lookahead: int) -> None:
        env = _get_env(control_timestep=0.01, n_steps_lookahead=n_steps_lookahead)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        sustains = note_traj.sustains
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_goal = np.zeros((n_steps_lookahead + 1, env.task.piano.n_keys + 1))

            t_start = i
            t_end = min(i + n_steps_lookahead + 1, len(notes))
            for j, t in enumerate(range(t_start, t_end)):
                keys = [note.key for note in notes[t]]
                expected_goal[j, keys] = 1.0
                expected_goal[j, -1] = sustains[t]

            actual_goal = timestep.observation["goal"]
            np.testing.assert_array_equal(actual_goal, expected_goal.ravel())

            # Check that the 0th goal is always the goal at the current timestep.
            expected_current = np.zeros((env.task.piano.n_keys + 1,))
            keys = [note.key for note in notes[i]]
            expected_current[keys] = 1.0
            expected_current[-1] = sustains[i]
            actual_current = timestep.observation["goal"][0 : env.task.piano.n_keys + 1]
            np.testing.assert_array_equal(actual_current, expected_current)

            timestep = env.step(zero_action)

            # In the `after_step` method, we cache the goal for the current timestep
            # to compute the reward. Let's check that it matches the expected goal.
            np.testing.assert_array_equal(expected_current, env.task._goal_current)

    @parameterized.parameters(
        self_actuated_piano.RewardType.NEGATIVE_L2,
        self_actuated_piano.RewardType.NEGATIVE_XENT,
    )
    def test_reward(self, reward_type: self_actuated_piano.RewardType) -> None:
        env = _get_env(reward_type=reward_type)
        action_spec = env.action_spec()
        timestep = env.reset()

        # The first timestep should have a None reward.
        self.assertIsNone(timestep.reward)

        while not timestep.last():
            random_ctrl = np.random.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
                size=action_spec.shape,
            ).astype(action_spec.dtype)
            timestep = env.step(random_ctrl)

            actual_reward = timestep.reward
            expected_reward = reward_type.get()(
                np.concatenate(
                    [env.task.piano.activation, env.task.piano.sustain_activation]
                ),
                env.task._goal_current,
            )
            self.assertEqual(actual_reward, expected_reward)


if __name__ == "__main__":
    absltest.main()

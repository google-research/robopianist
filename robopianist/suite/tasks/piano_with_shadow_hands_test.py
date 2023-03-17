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

"""Tests for piano_with_shadow_hands_test.py."""

import itertools
from typing import Optional

import numpy as np
from absl.testing import absltest, parameterized
from dm_control import composer
from mujoco_utils import spec_utils
from note_seq.protobuf import music_pb2

from robopianist.music import midi_file
from robopianist.suite.tasks import piano_with_shadow_hands


def _get_test_midi(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # C6 for 2 dts.
    seq.notes.add(
        start_time=0.0,
        end_time=2 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,  # Right hand index.
    )
    # G5 for 1 dt.
    seq.notes.add(
        start_time=2 * dt,
        end_time=3 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("G5"),
        part=0,  # Left hand thumb.
    )

    seq.total_time = 3 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_env(
    control_timestep: float = 0.01,
    n_steps_lookahead: int = 0,
    n_seconds_lookahead: Optional[float] = None,
    wrong_press_termination: bool = False,
    disable_fingering_reward: bool = False,
) -> composer.Environment:
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=_get_test_midi(dt=control_timestep),
        n_steps_lookahead=n_steps_lookahead,
        n_seconds_lookahead=n_seconds_lookahead,
        control_timestep=control_timestep,
        wrong_press_termination=wrong_press_termination,
        change_color_on_activation=True,
        disable_fingering_reward=disable_fingering_reward,
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class PianoWithShadowHandsTest(parameterized.TestCase):
    @parameterized.parameters(True, False)
    def test_observables(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        timestep = env.reset()

        # Piano observables.
        self.assertIn("piano/state", timestep.observation)
        self.assertIn("piano/sustain_state", timestep.observation)

        # Goal observables.
        self.assertIn("goal", timestep.observation)
        if disable_fingering_reward:
            self.assertNotIn("fingering", timestep.observation)
        else:
            self.assertIn("fingering", timestep.observation)

        # Hand observables.
        for name in ["rh_shadow_hand", "lh_shadow_hand"]:
            self.assertIn(f"{name}/joints_pos", timestep.observation)
            self.assertIn(f"{name}/position", timestep.observation)

    def test_action_spec(self) -> None:
        env = _get_env()
        rh_action_spec = env.task.right_hand.action_spec(env.physics)
        lh_action_spec = env.task.left_hand.action_spec(env.physics)
        combined_spec = spec_utils.merge_specs([rh_action_spec, lh_action_spec])
        actual_shape = env.action_spec().shape[0] - 1  # Don't include sustain pedal.
        expected_shape = combined_spec.shape[0]
        self.assertEqual(actual_shape, expected_shape)

        right_action = np.random.uniform(
            low=rh_action_spec.minimum, high=rh_action_spec.maximum
        ).astype(rh_action_spec.dtype)
        left_action = np.random.uniform(
            low=lh_action_spec.minimum, high=lh_action_spec.maximum
        ).astype(lh_action_spec.dtype)
        action = np.concatenate([right_action, left_action, [0]])
        env.task.before_step(env.physics, action, env.random_state)

        actual_rh_action = env.physics.bind(env.task.right_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_rh_action, right_action)
        actual_lh_action = env.physics.bind(env.task.left_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_lh_action, left_action)

    def test_termination_and_discount(self) -> None:
        env = _get_env()
        action_spec = env.action_spec()
        env.reset()

        # With a dt of 0.01 and a 3 dt long midi, the episode should end after 4 steps.
        zero_action = np.zeros(action_spec.shape)
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

    @parameterized.parameters(itertools.product([0.01, 0.05, 0.1], [0, 0.01, 0.1, 1]))
    def test_n_seconds_lookahead(
        self, control_timestep: float, n_seconds_lookahead: float
    ) -> None:
        env = _get_env(
            control_timestep=control_timestep, n_seconds_lookahead=n_seconds_lookahead
        )

        actual_n_steps_lookahead = env.task._n_steps_lookahead
        expected_n_steps_lookahead = int(
            np.ceil(n_seconds_lookahead / control_timestep)
        )
        self.assertEqual(actual_n_steps_lookahead, expected_n_steps_lookahead)

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

    def test_fingering_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_fingering = np.zeros((2, 5))
            idxs = [note.fingering for note in notes[i]]
            rh_idxs = [idx for idx in idxs if idx < 5]
            lh_idxs = [idx - 5 for idx in idxs if idx >= 5]
            expected_fingering[0, rh_idxs] = 1.0
            expected_fingering[1, lh_idxs] = 1.0

            actual_fingering = timestep.observation["fingering"]
            np.testing.assert_array_equal(actual_fingering, expected_fingering.ravel())

            timestep = env.step(zero_action)

            # In the `after_step` method, we cache the fingering information for the
            # current timestep to compute the reward. Let's check that it matches the
            # expected one.
            actual_rh_current = [r[1] for r in env.task._rh_keys_current]
            np.testing.assert_array_equal(rh_idxs, actual_rh_current)
            actual_lh_current = [r[1] for r in env.task._lh_keys_current]
            np.testing.assert_array_equal(lh_idxs, actual_lh_current)

    def test_failure_termination(self) -> None:
        env = _get_env(wrong_press_termination=True)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        # Simulate a wrong press by applying a generalized force on all the keys.
        env.physics.bind(env.task.piano.joints).qfrc_applied = 3.0

        # The episode should terminate in a single step.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # Failure, so discount should be 0.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 0.0)

    @absltest.skip("this observable is disabled")
    def test_steps_left_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)

        timestep = env.reset()
        self.assertEqual(timestep.observation["steps_left"], 1.0)

        for i in range(3):
            timestep = env.step(zero_action)
            self.assertAlmostEqual(
                timestep.observation["steps_left"], 1.0 - (i + 1) / 3
            )

    @parameterized.parameters(True, False)
    def test_fingering_reward_presence(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        env.step(zero_action)
        reward_terms = env.task.reward_fn.reward_terms

        if disable_fingering_reward:
            self.assertNotIn("fingering_reward", reward_terms)
        else:
            self.assertIn("fingering_reward", reward_terms)

    # TODO(kevin): Add unit tests for individual reward components.
    # TODO(kevin): Add unit tests for augmentation / midi selection.


if __name__ == "__main__":
    absltest.main()

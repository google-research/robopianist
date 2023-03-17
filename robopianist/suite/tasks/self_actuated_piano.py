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

"""A self-actuated piano that must learn to play a MIDI file."""

import enum
from typing import Callable, Optional, Sequence

import numpy as np
from dm_control import mjcf
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import spec_utils

from robopianist.models.arenas import stage
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base

# For numerical stability.
_EPS = 1e-6

RewardFn = Callable[[np.ndarray, np.ndarray], float]


def negative_binary_cross_entropy(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Computes the negative binary cross entropy between predictions and targets."""
    assert predictions.shape == targets.shape
    assert predictions.ndim >= 1
    log_p = np.log(predictions + _EPS)
    log_1_minus_p = np.log(1 - predictions + _EPS)
    return np.sum(targets * log_p + (1 - targets) * log_1_minus_p)


def negative_l2_distance(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Computes the negative L2 distance between predictions and targets."""
    assert predictions.shape == targets.shape
    assert predictions.ndim >= 1
    return -np.sqrt(np.sum((predictions - targets) ** 2))


class RewardType(enum.Enum):
    NEGATIVE_XENT = "negative_xent"
    NEGATIVE_L2 = "negative_l2"

    def get(self) -> RewardFn:
        if self == RewardType.NEGATIVE_XENT:
            return negative_binary_cross_entropy
        elif self == RewardType.NEGATIVE_L2:
            return negative_l2_distance
        else:
            raise ValueError(f"Invalid reward type: {self}")


class SelfActuatedPiano(base.PianoOnlyTask):
    """Task where a piano self-actuates to play a MIDI file."""

    def __init__(
        self,
        midi: midi_file.MidiFile,
        n_steps_lookahead: int = 0,
        trim_silence: bool = False,
        reward_type: RewardType = RewardType.NEGATIVE_L2,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        **kwargs,
    ) -> None:
        """Task constructor.

        Args:
            midi: A `MidiFile` object.
            n_steps_lookahead: Number of timesteps to look ahead when computing the
                goal state.
            trim_silence: If True, remove initial and final timesteps without any notes.
            reward_type: Reward function to use for the key press reward.
            augmentations: A list of `Variation` objects that will be applied to the
                MIDI file at the beginning of each episode. If None, no augmentations
                will be applied.
        """
        super().__init__(arena=stage.Stage(), add_piano_actuators=True, **kwargs)

        self._midi = midi
        self._initial_midi = midi
        self._n_steps_lookahead = n_steps_lookahead
        self._trim_silence = trim_silence
        self._key_press_reward = reward_type.get()
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
        )
        self._augmentations = augmentations

        self._reset_quantities_at_episode_init()
        self._reset_trajectory()  # Important: call before adding observables.
        self._add_observables()

    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False

    def _maybe_change_midi(self, random_state: np.random.RandomState) -> None:
        if self._augmentations is not None:
            midi = self._initial_midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._midi = midi
            self._reset_trajectory()

    def _reset_trajectory(self) -> None:
        note_traj = midi_file.NoteTrajectory.from_midi(
            self._midi, self.control_timestep
        )
        if self._trim_silence:
            note_traj.trim_silence()
        self._notes = note_traj.notes
        self._sustains = note_traj.sustains

    # Composer methods.

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del physics  # Unused.
        self._maybe_change_midi(random_state)
        self._reset_quantities_at_episode_init()

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        # Note that with a self-actuated piano, we don't need to separately apply the
        # sustain action.
        self.piano.apply_action(physics, action, random_state)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del physics, random_state  # Unused.
        self._t_idx += 1
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1

        # We need to save the goal state for the current timestep because observable
        # callables are called _before_ the reward is computed. Otherwise, we'd be off
        # by one timestep when computing the reward.
        # NOTE(kevin): The reason we don't need a `copy()` here is because
        # self._goal_state gets new memory allocated to it every time we call
        # `self._update_goal_state()`. For peace of mind, we have a unit test for this
        # in `play_midi_test.py`.
        self._goal_current = self._goal_state[0]

    def get_reward(self, physics: mjcf.Physics) -> float:
        return self._reward_fn.compute(physics)

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics  # Unused.
        return self._should_terminate

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        keys_spec = spec_utils.create_action_spec(physics, self.piano.actuators)
        sustain_spec = specs.BoundedArray(
            shape=(1,),
            dtype=keys_spec.dtype,
            minimum=[0.0],
            maximum=[1.0],
            name="sustain",
        )
        return spec_utils.merge_specs([keys_spec, sustain_spec])

    # Other.

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        return self._reward_fn

    # Helper methods.

    def _compute_key_press_reward(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return self._key_press_reward(
            np.concatenate([self.piano.activation, self.piano.sustain_activation]),
            self._goal_current,
        )

    def _update_goal_state(self) -> None:
        # Observable callables get called after `after_step` but before
        # `should_terminate_episode`. Since we increment `self._t_idx` in `after_step`,
        # we need to guard against out of bounds indexing. Note that the goal state
        # does not matter at this point since we are terminating the episode and this
        # update is usually meant for the next timestep.
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            self._goal_state[i, keys] = 1.0
            self._goal_state[i, -1] = self._sustains[t]

    def _add_observables(self) -> None:
        # This returns the current state of the piano keys.
        self.piano.observables.activation.enabled = True
        self.piano.observables.sustain_activation.enabled = True

        # This returns the goal state for the current timestep and n steps ahead.
        def _get_goal_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_goal_state()
            return self._goal_state.ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

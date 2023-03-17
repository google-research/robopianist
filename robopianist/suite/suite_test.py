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

"""Tests for robopianist.suite."""

import numpy as np
from absl.testing import absltest, parameterized

from robopianist import suite

_SEED = 12345
_NUM_EPISODES = 1
_NUM_STEPS_PER_EPISODE = 10


class RoboPianistSuiteTest(parameterized.TestCase):
    """Tests for all registered tasks in robopianist.suite."""

    def _validate_observation(self, observation, observation_spec):
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    @parameterized.parameters(*suite.ALL)
    def test_task_runs(self, environment_name: str) -> None:
        """Tests task loading and observation spec validity."""
        env = suite.load(environment_name, seed=_SEED)
        random_state = np.random.RandomState(_SEED)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        self.assertTrue(np.all(np.isfinite(action_spec.minimum)))
        self.assertTrue(np.all(np.isfinite(action_spec.maximum)))

        for _ in range(_NUM_EPISODES):
            timestep = env.reset()
            for _ in range(_NUM_STEPS_PER_EPISODE):
                self._validate_observation(timestep.observation, observation_spec)
                if timestep.first():
                    self.assertIsNone(timestep.reward)
                    self.assertIsNone(timestep.discount)
                action = random_state.uniform(
                    action_spec.minimum, action_spec.maximum, size=action_spec.shape
                ).astype(action_spec.dtype)
                timestep = env.step(action)


if __name__ == "__main__":
    absltest.main()

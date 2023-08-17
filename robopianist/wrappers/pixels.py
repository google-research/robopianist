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

"""A wrapper for adding pixels to the observation."""


import collections
from typing import Any, Dict, Optional

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper


class PixelWrapper(EnvironmentWrapper):
    """Adds pixel observations to the observation spec."""

    def __init__(
        self,
        environment: dm_env.Environment,
        render_kwargs: Optional[Dict[str, Any]] = None,
        observation_key: str = "pixels",
    ) -> None:
        super().__init__(environment)

        self._render_kwargs = render_kwargs or {}
        self._observation_key = observation_key

        # Update the observation spec.
        self._wrapped_observation_spec = self._environment.observation_spec()
        self._observation_spec = collections.OrderedDict()
        self._observation_spec.update(self._wrapped_observation_spec)
        pixels = self._environment.physics.render(**self._render_kwargs)
        pixels_spec = specs.Array(
            shape=pixels.shape, dtype=pixels.dtype, name=self._observation_key
        )
        self._observation_spec[observation_key] = pixels_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return self._add_pixel_observation(timestep)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return self._add_pixel_observation(timestep)

    def _add_pixel_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        pixels = self._environment.physics.render(**self._render_kwargs)
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{self._observation_key: pixels}
            )
        )

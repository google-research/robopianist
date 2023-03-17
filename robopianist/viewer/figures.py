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

from typing import Optional

from robopianist.viewer import views


class MujocoFigureModelWithRuntime(views.MujocoFigureModel):
    """Base class for figures that need access to the runtime."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._runtime = None
        self._on_episode_begin_callbacks = [self.reset]

    def set_runtime(self, instance) -> None:
        if self._runtime is not None:
            for callback in self._on_episode_begin_callbacks:
                self._runtime.on_episode_begin -= callback
        self._runtime = instance
        if self._runtime:
            for callback in self._on_episode_begin_callbacks:
                self._runtime.on_episode_begin += callback


class RewardFigure(MujocoFigureModelWithRuntime):
    """Plot the total reward over time."""

    def __init__(self, pause, **kwargs) -> None:
        super().__init__(**kwargs)

        self._pause = pause
        self._series = views.TimeSeries()

        self._on_episode_begin_callbacks.append(self.reset_series)

    def configure_figure(self) -> None:
        self._figure.title = "Reward"
        self._figure.xlabel = "Timestep"

    def get_time_series(self) -> Optional[views.TimeSeries]:
        if self._runtime is None or self._pause.value:
            return None

        reward = self._runtime._time_step.reward
        self._series.add(reward)

        return self._series

    def reset_series(self) -> None:
        self._series.clear()


class RewardTermsFigure(MujocoFigureModelWithRuntime):
    """Plot the different reward terms over time."""

    def __init__(self, pause, **kwargs) -> None:
        super().__init__(**kwargs)

        self._pause = pause
        self._series = views.TimeSeries()

        self._on_episode_begin_callbacks.append(self.reset_series)

    def configure_figure(self) -> None:
        self._figure.title = "Reward"
        self._figure.xlabel = "Timestep"

    def get_time_series(self) -> Optional[views.TimeSeries]:
        if self._runtime is None or self._pause.value:
            return None

        reward = self._runtime._time_step.reward

        if not hasattr(self._runtime._environment.task, "reward_fn"):
            self._series.add(reward)
        else:
            reward_fn = self._runtime._environment.task.reward_fn
            reward_dict = {k: v for k, v in reward_fn.reward_terms.items()}
            reward_dict["total"] = reward  # Also log the total reward.
            self._series.add_dict(reward_dict)

        return self._series

    def reset_series(self) -> None:
        self._series.clear()

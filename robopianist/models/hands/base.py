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

import abc
import enum
from typing import Sequence

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from mujoco_utils import types


@enum.unique
class HandSide(enum.Enum):
    """Which hand side is being modeled."""

    LEFT = enum.auto()
    RIGHT = enum.auto()


class Hand(composer.Entity, abc.ABC):
    """Base composer class for dexterous hands."""

    def _build_observables(self) -> "HandObservables":
        return HandObservables(self)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def hand_side(self) -> HandSide:
        ...

    @property
    @abc.abstractmethod
    def root_body(self) -> types.MjcfElement:
        ...

    @property
    @abc.abstractmethod
    def joints(self) -> Sequence[types.MjcfElement]:
        ...

    @property
    @abc.abstractmethod
    def actuators(self) -> Sequence[types.MjcfElement]:
        ...

    @property
    @abc.abstractmethod
    def fingertip_sites(self) -> Sequence[types.MjcfElement]:
        ...


class HandObservables(composer.Observables):
    """Base class for dexterous hand observables."""

    _entity: Hand

    @composer.observable
    def joints_pos(self):
        """Returns the joint positions."""
        return observable.MJCFFeature("qpos", self._entity.joints)

    @composer.observable
    def joints_pos_cos_sin(self):
        """Returns the joint positions encoded as (cos, sin) pairs.

        This has twice as many dimensions as the raw joint positions.
        """

        def _get_joint_angles(physics: mjcf.Physics) -> np.ndarray:
            qpos = physics.bind(self._entity.joints).qpos
            return np.hstack([np.cos(qpos), np.sin(qpos)])

        return observable.Generic(raw_observation_callable=_get_joint_angles)

    @composer.observable
    def joints_vel(self):
        """Returns the joint velocities."""
        return observable.MJCFFeature("qvel", self._entity.joints)

    @composer.observable
    def joints_torque(self) -> observable.Generic:
        """Returns the joint torques."""

        def _get_joint_torques(physics: mjcf.Physics) -> np.ndarray:
            # We only care about torques acting on each joint's axis of rotation, so we
            # project them.
            torques = physics.bind(self._entity.joint_torque_sensors).sensordata
            joint_axes = physics.bind(self._entity.joints).axis
            return np.einsum("ij,ij->i", torques.reshape(-1, 3), joint_axes)

        return observable.Generic(raw_observation_callable=_get_joint_torques)

    @composer.observable
    def position(self):
        """Returns the position of the hand's root body in the world frame."""
        return observable.MJCFFeature("xpos", self._entity.root_body)

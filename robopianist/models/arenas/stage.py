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

"""Suite arenas."""

from mujoco_utils import composer_utils


class Stage(composer_utils.Arena):
    """A custom arena with a ground plane, lights and a starry night sky."""

    def _build(self, name: str = "stage") -> None:
        super()._build(name=name)

        # Change free camera settings.
        self._mjcf_root.statistic.extent = 0.6
        self._mjcf_root.statistic.center = (0.2, 0, 0.3)
        getattr(self._mjcf_root.visual, "global").azimuth = 180
        getattr(self._mjcf_root.visual, "global").elevation = -50

        self._mjcf_root.visual.map.stiffness = 400
        self._mjcf_root.visual.scale.forcewidth = 0.04
        self._mjcf_root.visual.scale.contactwidth = 0.2
        self._mjcf_root.visual.scale.contactheight = 0.03

        # Lights.
        self._mjcf_root.worldbody.add("light", pos=(0, 0, 1))
        self._mjcf_root.worldbody.add(
            "light", pos=(0.3, 0, 1), dir=(0, 0, -1), directional=False
        )

        # Dark checkerboard floor.
        self._mjcf_root.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=[0.1, 0.1, 0.1],
            rgb2=[0.2, 0.2, 0.2],
        )
        self._mjcf_root.asset.add(
            "material",
            name="grid",
            texture="grid",
            texrepeat=(1, 1),
            texuniform=True,
            reflectance=0.2,
        )
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            type="plane",
            size=(1, 1, 0.05),
            material="grid",
            contype=0,
            conaffinity=0,
        )

        # Starry night sky.
        self._mjcf_root.asset.add(
            "texture",
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=[0.2, 0.2, 0.2],
            rgb2=[0.0, 0.0, 0.0],
            width=800,
            height=800,
            mark="random",
            markrgb=[1, 1, 1],
        )

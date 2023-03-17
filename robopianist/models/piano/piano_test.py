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

"""Tests for piano.py."""

from absl.testing import absltest
from dm_control import mjcf

from robopianist.models.piano import piano
from robopianist.models.piano import piano_constants as consts


class PianoTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        robot = piano.Piano()
        physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
        for _ in range(100):
            physics.step()

    def test_set_name(self) -> None:
        robot = piano.Piano(name="mozart")
        self.assertEqual(robot.mjcf_model.model, "mozart")

    def test_joints(self) -> None:
        robot = piano.Piano()
        self.assertEqual(len(robot.joints), consts.NUM_KEYS)
        for joint in robot.joints:
            self.assertEqual(joint.tag, "joint")

    def test_keys(self) -> None:
        robot = piano.Piano()
        self.assertEqual(len(robot.keys), consts.NUM_KEYS)
        for key in robot.keys:
            self.assertEqual(key.tag, "body")

    def test_sorted(self) -> None:
        robot = piano.Piano()
        for i in range(consts.NUM_KEYS - 1):
            self.assertLess(
                int(robot.keys[i].name.split("_")[-1]),
                int(robot.keys[i + 1].name.split("_")[-1]),
            )
            self.assertLess(
                int(robot._sites[i].name.split("_")[-1]),
                int(robot._sites[i + 1].name.split("_")[-1]),
            )
            self.assertLess(
                int(robot._key_geoms[i].name.split("_")[-1]),
                int(robot._key_geoms[i + 1].name.split("_")[-1]),
            )
            self.assertLess(
                int(robot.joints[i].name.split("_")[-1]),
                int(robot.joints[i + 1].name.split("_")[-1]),
            )


if __name__ == "__main__":
    absltest.main()

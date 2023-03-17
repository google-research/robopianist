# Copyright 2017 The dm_control Authors.
# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Suite environments viewer package."""


from robopianist.viewer import application


def launch(
    environment_loader, policy=None, title="RoboPianist", width=1024, height=768
) -> None:
    app = application.Application(title=title, width=width, height=height)
    app.launch(environment_loader=environment_loader, policy=policy)

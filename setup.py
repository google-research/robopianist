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

import re
from pathlib import Path

from setuptools import find_packages, setup

_here = Path(__file__).resolve().parent

name = "robopianist"

# Reference: https://github.com/patrick-kidger/equinox/blob/main/setup.py
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")


with open(_here / "README.md", "r") as f:
    readme = f.read()

core_requirements = [
    "dm_control>=1.0.9",
    "dm_env_wrappers>=0.0.11",
    "mujoco>=2.3.1",
    "mujoco_utils>=0.0.6",
    "note_seq>=0.0.5",
    "pretty_midi>=0.2.10",
    "pyaudio>=0.2.12",
    "pyfluidsynth>=1.3.2",
    "scikit-learn",
    "termcolor",
    "tqdm",
]

test_requirements = [
    "absl-py",
    "pytest-xdist",
]

dev_requirements = [
    "black",
    "ruff",
    "mypy",
] + test_requirements

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

author = "Kevin Zakka"

author_email = "kevinarmandzakka@gmail.com"

description = "A benchmark for high-dimensional robot control"

keywords = "reinforcement-learning mujoco bimanual dexterous-manipulation piano"

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=keywords,
    url=f"https://github.com/google-research/{name}",
    license="Apache License 2.0",
    license_files=("LICENSE",),
    packages=find_packages(exclude=["examples"]),
    python_requires=">=3.8",
    install_requires=core_requirements,
    include_package_data=True,
    classifiers=classifiers,
    extras_require={
        "test": test_requirements,
        "dev": dev_requirements,
    },
    zip_safe=False,
    entry_points={"console_scripts": [f"{name}={name}.cli:main"]},
)

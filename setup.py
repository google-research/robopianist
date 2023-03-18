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

import fnmatch
import os
import re
import shutil
from distutils import cmd
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

_here = Path(__file__).resolve().parent

_menagerie_dir = _here / "third_party" / "mujoco_menagerie"
_hand_dir = _here / "robopianist" / "models" / "hands" / "third_party"

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
    "mujoco_utils>=0.0.5",
    "note_seq == 0.0.3",
    "pretty_midi>=0.2.10",
    "protobuf==3.20.0",
    "pyaudio >= 0.2.12",
    "pyfluidsynth >= 1.3.2",
    "scikit-learn",
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
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

author = "Kevin Zakka"

author_email = "kevinarmandzakka@gmail.com"

description = "A benchmark for high-dimensional robot control"

keywords = "reinforcement-learning mujoco bimanual dexterous-manipulation piano"


# Reference: https://github.com/deepmind/dm_control/blob/main/setup.py
def find_data_files(package_dir, patterns, excludes=()):
    """Recursively finds files whose names match the given shell patterns."""
    paths = set()

    def is_excluded(s):
        for exclude in excludes:
            if fnmatch.fnmatch(s, exclude):
                return True
        return False

    for directory, _, filenames in os.walk(package_dir):
        if is_excluded(directory):
            continue
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # NB: paths must be relative to the package directory.
                relative_dirpath = os.path.relpath(directory, package_dir)
                full_path = os.path.join(relative_dirpath, filename)
                if not is_excluded(full_path):
                    paths.add(full_path)

    return list(paths)


class _CopyMenagerieModels(cmd.Command):
    """Copy the menagerie models to robopianist/models/hands/third_party/."""

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        if not _menagerie_dir.exists() or not list(_menagerie_dir.iterdir()):
            raise RuntimeError(
                "The menagerie directory is empty. Please run git submodule init &&"
                "git submodule update to download the menagerie models."
            )
        if _hand_dir.exists():
            shutil.rmtree(_hand_dir)
        shutil.copytree(_menagerie_dir / "shadow_hand", _hand_dir / "shadow_hand")


class _BuildExt(build_ext):
    """Copy menagerie models in build_ext stage."""

    def run(self) -> None:
        self.run_command("copy_menagerie_models")
        build_ext.run(self)


class _BuildPy(build_py):
    """Copy menagerie models in build_py stage."""

    def run(self) -> None:
        self.run_command("copy_menagerie_models")
        build_py.run(self)


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
    package_data={
        name: find_data_files(
            package_dir=name,
            patterns=["*.png", "*.xml", "*.typed", ".proto"],
            excludes=[],
        ),
    },
    python_requires=">=3.8, <3.11",
    install_requires=core_requirements,
    classifiers=classifiers,
    extras_require={
        "test": test_requirements,
        "dev": dev_requirements,
    },
    cmdclass={
        "build_ext": _BuildExt,
        "build_py": _BuildPy,
        "copy_menagerie_models": _CopyMenagerieModels,
    },
)

# RoboPianist: A Benchmark for High-Dimensional Robot Control

[![build][tests-badge]][tests]
[![docs][docs-badge]][docs]
[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]

[tests-badge]: https://github.com/google-research/robopianist/actions/workflows/ci.yml/badge.svg
[docs-badge]: https://github.com/google-research/robopianist/actions/workflows/docs.yml/badge.svg
[tests]: https://github.com/google-research/robopianist/actions/workflows/ci.yml
[docs]: https://google-research.github.io/robopianist/
[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/robopianist
[pypi-badge]: https://badge.fury.io/py/robopianist.svg
[pypi]: https://pypi.org/project/robopianist/

![RoboPianist teaser image](./docs/teaser1x3.jpeg)

RoboPianist is a new benchmarking suite for high-dimensional control, targeted at testing high spatial and temporal precision, coordination, and planning, all with an underactuated system frequently making-and-breaking contacts. The proposed challenge is *mastering the piano* through bi-manual dexterity, using a pair of simulated anthropomorphic robot hands.

This codebase contains software and tasks for the benchmark, and is powered by [MuJoCo](https://mujoco.org/).

- [Getting Started](#getting-started)
- [Installation](#installation)
  - [Install from source](#install-from-source)
  - [Install from PyPI](#install-from-pypi)
  - [Optional: Download additional soundfonts](#optional-download-additional-soundfonts)
- [MIDI Dataset](#midi-dataset)
- [CLI](#cli)
- [Contributing](#contributing)
- [FAQ](#faq)
- [Citing RoboPianist](#citing-robopianist)
- [Acknowledgements](#acknowledgements)
- [License and Disclaimer](#license-and-disclaimer)

## Getting Started

We've created an introductory [Colab](https://colab.research.google.com/github/google-research/robopianist/blob/main/tutorial.ipynb) notebook that demonstrates how to use RoboPianist. It includes code for loading and customizing a piano playing task, and a demonstration of a pretrained policy playing a short snippet of *Twinkle Twinkle Little Star*. Click the button below to get started!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/robopianist/blob/main/tutorial.ipynb)

## Installation

RoboPianist is supported on both Linux and macOS and can be installed with Python 3.8 up to 3.10. We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environment.

**3.11 will be supported once the numba team resolves [#8304](https://github.com/numba/numba/issues/8304).**

### Install from source

The recommended way to install this package is from source. Start by cloning the repository:

```bash
git clone https://github.com/google-research/robopianist.git && cd robopianist
```

Next, install the prerequisite dependencies:

```bash
git submodule init && git submodule update
bash scripts/install_deps.sh
```

Finally, create a new conda environment and install RoboPianist in editable mode:

```bash
conda create -n pianist python=3.10
conda activate pianist

pip install -e ".[dev]"
```

To test your installation, run `make test` and verify that all tests pass.

### Install from PyPI

First, install the prerequisite dependencies:

```bash
bash <(curl -s https://raw.githubusercontent.com/google-research/robopianist/main/scripts/install_deps.sh) --no-soundfonts
```

Next, create a new conda environment and install RoboPianist:

```bash
conda create -n pianist python=3.10
conda activate pianist

pip install --upgrade robopianist
```

### Optional: Download additional soundfonts

We recommend installing additional soundfonts to improve the quality of the synthesized audio. You can easily do this using the RoboPianist CLI:

```bash
robopianist soundfont --download
```

For more soundfont-related commands, see [docs/soundfonts.md](docs/soundfonts.md).

## MIDI Dataset

The PIG dataset cannot be redistributed on GitHub due to licensing restrictions. See [docs/dataset](docs/dataset.md) for instructions on where to download it and how to preprocess it.

## CLI

RoboPianist comes with a command line interface (CLI) that can be used to download additional soundfonts, play MIDI files, preprocess the PIG dataset, and more. For more information, see [docs/cli.md](docs/cli.md).

## Contributing

We welcome contributions to RoboPianist. Please see [docs/contributing.md](docs/contributing.md) for more information.

## FAQ

See [docs/faq.md](docs/faq.md) for a list of frequently asked questions.

## Citing RoboPianist

If you use RoboPianist in your work, please use the following citation:

```bibtex
@software{zakka2023robopianist,
  author = {Zakka, Kevin and Smith, Laura and Gileadi, Nimrod and Howell, Taylor and Peng, Xue Bin and Singh, Sumeet and Tassa, Yuval and Florence, Pete and Zeng, Andy and Abbeel, Pieter},
  title = {{RoboPianist: A Benchmark for High-Dimensional Robot Control}},
  url = {https://github.com/google-research/robopianist},
  year = {2023},
}
```

## Acknowledgements

We would like to thank the following people for making this project possible:

- [Philipp Wu](https://www.linkedin.com/in/wuphilipp/) and [Mohit Shridhar](https://mohitshridhar.com/) for being a constant source of inspiration and support.
- [Ilya Kostrikov](https://www.kostrikov.xyz/) for constantly raising the bar for RL engineering and for invaluable debugging help.
- The [Magenta](https://magenta.tensorflow.org/) team for helpful pointers and feedback.
- The [MuJoCo](https://mujoco.org/) team for the development of the MuJoCo physics engine and their support throughout the project.

## License and Disclaimer

[MuJoco Menagerie](https://github.com/deepmind/mujoco_menagerie)'s license can be found [here](https://github.com/deepmind/mujoco_menagerie/blob/main/LICENSE). Soundfont licensing information can be found [here](docs/soundfonts.md). MIDI licensing information can be found [here](docs/dataset). All other code is licensed under an [Apache-2.0 License](LICENSE).

This is not an officially supported Google product.

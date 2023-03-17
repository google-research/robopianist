# RoboPianist: A Benchmark for High-Dimensional Robot Control

Software and tasks for high-dimensional robot control, powered by [MuJoCo](https://mujoco.org/).

![RoboPianist teaser image](./docs/teaser1x3.jpeg)

## Installation

1. `git clone https://github.com/google-research/robopianist.git`
2. `bash scripts/install_deps.sh`
3. `conda create -n pianist python=3.10`
4. `conda activate pianist`
5. `pip install -e ".[dev]"`
6. `git submodule init && git submodule update`
7. `make test`
8. Optional: `bash scripts/get_soundfonts.sh` to download additional soundfonts

## License and Disclaimer

[MuJoco Menagerie](https://github.com/deepmind/mujoco_menagerie)'s license can be found [here](https://github.com/deepmind/mujoco_menagerie/blob/main/LICENSE). Soundfont licensing information can be found [here](docs/soundfonts.md). MIDI licensing information can be found [here](docs/dataset). All other code is licensed under an [Apache-2.0 License](LICENSE).

This is not an officially supported Google product.

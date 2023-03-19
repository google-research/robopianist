# Frequently Asked Questions

- [Can I use my own custom soundfont?](#can-i-use-my-own-custom-soundfont)
- [I get a `ImportError("Couldn't find the FluidSynth library.")`](#i-get-a-importerrorcouldnt-find-the-fluidsynth-library)

## Can I use my own custom soundfont?

You are free to use a soundfont of your choosing, just make sure to update `SF2_PATH` in [`robopianist/__init__.py`](robopianist/__init__.py) to point to its location. Note only `.sf2` soundfonts are supported.

## I get a `ImportError("Couldn't find the FluidSynth library.")`

See [this stackoverflow answer](https://stackoverflow.com/a/75339618) for a solution.

# Frequently Asked Questions

- [Can I use my own custom soundfont?](#can-i-use-my-own-custom-soundfont)
- [I am getting lots of `DeprecationWarning: Call to deprecated create function FieldDescriptor()` warnings. What should I do?](#i-am-getting-lots-of-deprecationwarning-call-to-deprecated-create-function-fielddescriptor-warnings-what-should-i-do)
- [I get a `ImportError("Couldn't find the FluidSynth library.")`](#i-get-a-importerrorcouldnt-find-the-fluidsynth-library)

## Can I use my own custom soundfont?

You are free to use a soundfont of your choosing, just make sure to update `SF2_PATH` in [`robopianist/__init__.py`](robopianist/__init__.py) to point to its location. Note only `.sf2` soundfonts are supported.

## I am getting lots of `DeprecationWarning: Call to deprecated create function FieldDescriptor()` warnings. What should I do?

To eliminate these warnings, you can install the latest version of `note_seq`. This will update the protobuf version so you need to re-downgrade to 3.20.0. To do so, run:

```bash
pip install --upgrade note_seq
pip install protobuf==3.20.0
```

## I get a `ImportError("Couldn't find the FluidSynth library.")`

See [this stackoverflow answer](https://stackoverflow.com/a/75339618) for a solution.

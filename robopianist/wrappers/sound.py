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

"""A wrapper for rendering videos with sound."""

import shutil
import subprocess
import wave
from pathlib import Path

import dm_env
from dm_env_wrappers import DmControlVideoWrapper

from robopianist import SF2_PATH
from robopianist.models.piano import midi_module
from robopianist.music import constants as consts
from robopianist.music import midi_message, synthesizer


class PianoSoundVideoWrapper(DmControlVideoWrapper):
    """Video rendering with sound from the piano keys."""

    def __init__(
        self,
        environment: dm_env.Environment,
        sf2_path: Path = SF2_PATH,
        sample_rate: int = consts.SAMPLING_RATE,
        **kwargs,
    ) -> None:
        # Check that this is an environment with a piano.
        if not hasattr(environment.task, "piano"):
            raise ValueError("PianoVideoWrapper only works with piano environments.")

        super().__init__(environment, **kwargs)

        self._midi_module: midi_module.MidiModule = environment.task.piano.midi_module
        self._sample_rate = sample_rate
        self._synth = synthesizer.Synthesizer(sf2_path, sample_rate)

    def _write_frames(self) -> None:
        super()._write_frames()

        midi_events = self._midi_module.get_all_midi_messages()

        # Exit if there are no MIDI events or if all events are sustain events.
        # Sustain only events cause white noise in the audio (which has shattered my
        # eardrums on more than one occasion).
        no_events = len(midi_events) == 0
        are_events_sustains = [
            isinstance(event, (midi_message.SustainOn, midi_message.SustainOff))
            for event in midi_events
        ]
        only_sustain = all(are_events_sustains) and len(midi_events) > 0
        if no_events or only_sustain:
            return

        # Synthesize waveform.
        waveform = self._synth.get_samples(midi_events)

        # Save waveform as mp3.
        waveform_name = self._record_dir / f"{self._counter:05d}.mp3"
        wf = wave.open(str(waveform_name), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self._sample_rate * self._playback_speed)
        wf.writeframes(waveform)  # type: ignore
        wf.close()

        # Make a copy of the MP4 so that FFMPEG can overwrite it.
        filename = self._record_dir / f"{self._counter:05d}.mp4"
        temp_filename = self._record_dir / "temp.mp4"
        shutil.copyfile(filename, temp_filename)
        filename.unlink()

        # Add the sound to the MP4 using FFMPEG, suppressing the output.
        # Reference: https://stackoverflow.com/a/11783474
        ret = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(temp_filename),
                "-i",
                str(waveform_name),
                "-map",
                "0",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-shortest",
                str(filename),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
        if ret.returncode != 0:
            print(f"FFMPEG failed to add sound to video {temp_filename}.")

        # Remove temporary files.
        temp_filename.unlink()
        waveform_name.unlink()

    def __del__(self) -> None:
        self._synth.stop()

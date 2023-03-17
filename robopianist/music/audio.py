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

"""Audio playback utils."""

import time

import numpy as np
import pyaudio

from robopianist.music import constants as consts


def play_sound(
    waveform: np.ndarray, sampling_rate: int = consts.SAMPLING_RATE, chunk: int = 1024
) -> None:
    """Play a waveform using PyAudio."""
    if waveform.dtype != np.int16:
        raise ValueError("waveform must be an np.int16 array.")

    # An iterator that yields chunks of audio data.
    def chunkifier():
        for i in range(0, len(waveform), chunk):
            yield waveform[i : i + chunk]

    audio_generator = chunkifier()

    def callback(in_data, frame_count, time_info, status):
        del in_data, frame_count, time_info, status
        return (next(audio_generator), pyaudio.paContinue)

    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sampling_rate,
        output=True,
        frames_per_buffer=chunk,
        stream_callback=callback,
    )

    try:
        stream.start_stream()
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Ctrl-C detected. Stopping playback.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

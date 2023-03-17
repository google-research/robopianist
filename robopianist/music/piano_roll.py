# Copyright 2022 The Magenta Authors.
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

"""Piano roll utilities.

This is a copy of note_seq's implementation, modified to also compute fingering
information in the returned `Pianoroll` object.
"""

import collections
import math

import numpy as np
import pretty_midi
from note_seq import constants, music_pb2

# The amount to upweight note-on events vs note-off events.
ONSET_UPWEIGHT = 5.0

# The size of the frame extension for onset event.
# Frames in [onset_frame-ONSET_WINDOW, onset_frame+ONSET_WINDOW]
# are considered to contain onset events.
ONSET_WINDOW = 1

Pianoroll = collections.namedtuple(  # pylint:disable=invalid-name
    "Pianoroll",
    [
        "active",
        "weights",
        "onsets",
        "onset_velocities",
        "active_velocities",
        "offsets",
        "control_changes",
        "fingerings",
    ],
)


def _unscale_velocity(velocity, scale, bias):
    unscaled = max(min(velocity, 1.0), 0) * scale + bias
    if math.isnan(unscaled):
        return 0
    return int(unscaled)


def sequence_to_pianoroll(
    sequence,
    frames_per_second,
    min_pitch,
    max_pitch,
    # pylint: disable=unused-argument
    min_velocity=constants.MIN_MIDI_VELOCITY,
    # pylint: enable=unused-argument
    max_velocity=constants.MAX_MIDI_VELOCITY,
    add_blank_frame_before_onset=False,
    onset_upweight=ONSET_UPWEIGHT,
    onset_window=ONSET_WINDOW,
    onset_length_ms=0,
    offset_length_ms=0,
    onset_mode="window",
    onset_delay_ms=0.0,
    min_frame_occupancy_for_label=0.0,
    onset_overlap=True,
):
    roll = np.zeros(
        (int(sequence.total_time * frames_per_second + 1), max_pitch - min_pitch + 1),
        dtype=np.float32,
    )

    roll_weights = np.ones_like(roll)

    onsets = np.zeros_like(roll)
    offsets = np.zeros_like(roll)

    control_changes = np.zeros(
        (int(sequence.total_time * frames_per_second + 1), 128), dtype=np.int32
    )

    fingerings = np.full_like(roll, -1)

    def frames_from_times(start_time, end_time):
        """Converts start/end times to start/end frames."""
        # Will round down because note may start or end in the middle of the frame.
        start_frame = int(start_time * frames_per_second)
        start_frame_occupancy = start_frame + 1 - start_time * frames_per_second
        # check for > 0.0 to avoid possible numerical issues
        if (
            min_frame_occupancy_for_label > 0.0
            and start_frame_occupancy < min_frame_occupancy_for_label
        ):
            start_frame += 1

        end_frame = int(math.ceil(end_time * frames_per_second))
        end_frame_occupancy = end_time * frames_per_second - start_frame - 1
        if (
            min_frame_occupancy_for_label > 0.0
            and end_frame_occupancy < min_frame_occupancy_for_label
        ):
            end_frame -= 1

        # Ensure that every note fills at least one frame.
        end_frame = max(start_frame + 1, end_frame)

        return start_frame, end_frame

    velocities_roll = np.zeros_like(roll, dtype=np.float32)

    for note in sorted(sequence.notes, key=lambda n: n.start_time):
        if note.pitch < min_pitch or note.pitch > max_pitch:
            continue
        start_frame, end_frame = frames_from_times(note.start_time, note.end_time)

        # label onset events. Use a window size of onset_window to account of
        # rounding issue in the start_frame computation.
        onset_start_time = note.start_time + onset_delay_ms / 1000.0
        onset_end_time = note.end_time + onset_delay_ms / 1000.0
        if onset_mode == "window":
            onset_start_frame_without_window, _ = frames_from_times(
                onset_start_time, onset_end_time
            )

            onset_start_frame = max(0, onset_start_frame_without_window - onset_window)
            onset_end_frame = min(
                onsets.shape[0], onset_start_frame_without_window + onset_window + 1
            )
        elif onset_mode == "length_ms":
            onset_end_time = min(
                onset_end_time, onset_start_time + onset_length_ms / 1000.0
            )
            onset_start_frame, onset_end_frame = frames_from_times(
                onset_start_time, onset_end_time
            )
        else:
            raise ValueError("Unknown onset mode: {}".format(onset_mode))

        # label offset events.
        offset_start_time = min(
            note.end_time, sequence.total_time - offset_length_ms / 1000.0
        )
        offset_end_time = offset_start_time + offset_length_ms / 1000.0
        offset_start_frame, offset_end_frame = frames_from_times(
            offset_start_time, offset_end_time
        )
        offset_end_frame = max(offset_end_frame, offset_start_frame + 1)

        if not onset_overlap:
            start_frame = onset_end_frame
            end_frame = max(start_frame + 1, end_frame)

        offsets[offset_start_frame:offset_end_frame, note.pitch - min_pitch] = 1.0
        onsets[onset_start_frame:onset_end_frame, note.pitch - min_pitch] = 1.0
        roll[start_frame:end_frame, note.pitch - min_pitch] = 1.0

        if note.velocity > max_velocity:
            raise ValueError(
                "Note velocity exceeds max velocity: %d > %d"
                % (note.velocity, max_velocity)
            )

        velocities_roll[start_frame:end_frame, note.pitch - min_pitch] = (
            note.velocity / max_velocity
        )
        roll_weights[
            onset_start_frame:onset_end_frame, note.pitch - min_pitch
        ] = onset_upweight
        roll_weights[onset_end_frame:end_frame, note.pitch - min_pitch] = [
            onset_upweight / x for x in range(1, end_frame - onset_end_frame + 1)
        ]
        if note.part is not None:
            fingerings[start_frame:end_frame, note.pitch - min_pitch] = note.part

        if add_blank_frame_before_onset:
            if start_frame > 0:
                roll[start_frame - 1, note.pitch - min_pitch] = 0.0
                roll_weights[start_frame - 1, note.pitch - min_pitch] = 1.0

    for cc in sequence.control_changes:
        frame, _ = frames_from_times(cc.time, 0)
        if frame < len(control_changes):
            control_changes[frame, cc.control_number] = cc.control_value + 1

    return Pianoroll(
        active=roll,
        weights=roll_weights,
        onsets=onsets,
        onset_velocities=velocities_roll * onsets,
        active_velocities=velocities_roll,
        offsets=offsets,
        control_changes=control_changes,
        fingerings=fingerings,
    )


def pianoroll_onsets_to_note_sequence(
    onsets,
    frames_per_second,
    note_duration_seconds=0.05,
    velocity=70,
    instrument=0,
    program=0,
    qpm=constants.DEFAULT_QUARTERS_PER_MINUTE,
    min_midi_pitch=constants.MIN_MIDI_PITCH,
    velocity_values=None,
    velocity_scale=80,
    velocity_bias=10,
):
    frame_length_seconds = 1 / frames_per_second

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().qpm = qpm
    sequence.ticks_per_quarter = constants.STANDARD_PPQ

    if velocity_values is None:
        velocity_values = velocity * np.ones_like(onsets, dtype=np.int32)

    for frame, pitch in zip(*np.nonzero(onsets)):
        start_time = frame * frame_length_seconds
        end_time = start_time + note_duration_seconds

        note = sequence.notes.add()
        note.start_time = start_time
        note.end_time = end_time
        note.pitch = pitch + min_midi_pitch
        note.velocity = _unscale_velocity(
            velocity_values[frame, pitch], scale=velocity_scale, bias=velocity_bias
        )
        note.instrument = instrument
        note.program = program

    sequence.total_time = len(onsets) * frame_length_seconds + note_duration_seconds
    if sequence.notes:
        assert sequence.total_time >= sequence.notes[-1].end_time

    return sequence


def sequence_to_valued_intervals(
    note_sequence,
    min_midi_pitch=constants.MIN_MIDI_PITCH,
    max_midi_pitch=constants.MAX_MIDI_PITCH,
    restrict_to_pitch=None,
):
    """Convert a NoteSequence to valued intervals.
    Value intervals are intended to be used with mir_eval metrics methods.
    Args:
      note_sequence: sequence to convert.
      min_midi_pitch: notes lower than this will be discarded.
      max_midi_pitch: notes higher than this will be discarded.
      restrict_to_pitch: notes that are not this pitch will be discarded.
    Returns:
      intervals: start and end times
      pitches: pitches in Hz.
      velocities: MIDI velocities.
    """
    intervals = []
    pitches = []
    velocities = []

    for note in note_sequence.notes:
        if restrict_to_pitch and restrict_to_pitch != note.pitch:
            continue
        if note.pitch < min_midi_pitch or note.pitch > max_midi_pitch:
            continue
        # mir_eval does not allow notes that start and end at the same time.
        if note.end_time == note.start_time:
            continue
        intervals.append((note.start_time, note.end_time))
        pitches.append(note.pitch)
        velocities.append(note.velocity)

    # Reshape intervals to ensure that the second dim is 2, even if the list is
    # of size 0. mir_eval functions will complain if intervals is not shaped
    # appropriately.
    intervals = np.array(intervals).reshape((-1, 2))
    pitches = np.array(pitches)
    pitches = pretty_midi.note_number_to_hz(pitches)
    velocities = np.array(velocities)
    return intervals, pitches, velocities

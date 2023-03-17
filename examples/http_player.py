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

"""An HTTP server that plays notes received through POST requests.

To try it out, start the server:
  python examples/http_player.py

Then send it a post request like:
  curl -X POST localhost:8080 -d 'ACTIVATION=[40,44]'
"""

import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np

from robopianist.music import midi_file, synthesizer
from robopianist.music.constants import NUM_KEYS

hostname = "localhost"
serverport = 8080

_ACTIVATION_RE = re.compile(r"^ACTIVATION=\[((\d+,)*(\d+)?)\]$")


class PianoServer(BaseHTTPRequestHandler):
    """An HTTP server that plays notes received through POST requests."""

    def __init__(self, *args, **kwargs):
        self._prev_activation = np.zeros(NUM_KEYS, dtype=bool)

        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        global _synth
        assert _synth is not None

        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

        for line in self.rfile:
            m = _ACTIVATION_RE.match(str(line, "utf-8"))
            if m:
                activation = np.zeros(NUM_KEYS, dtype=bool)
                if m.group(1):
                    active = [int(d) for d in m.group(1).split(",")]
                    for key_id in active:
                        if key_id < NUM_KEYS:
                            activation[key_id] = True
                        else:
                            print(f"Invalid key id: {key_id}")

                state_change = activation ^ self._prev_activation

                # Note on events.
                for key_id in np.flatnonzero(state_change & ~self._prev_activation):
                    _synth.note_on(
                        midi_file.key_number_to_midi_number(key_id),
                        127,
                    )

                # Note off events.
                for key_id in np.flatnonzero(state_change & ~activation):
                    _synth.note_off(midi_file.key_number_to_midi_number(key_id))

                # Update state.
                self._prev_activation = activation.copy()

            break

    def log_request(self, request):
        del request  # Unused.


_synth: Optional[synthesizer.Synthesizer] = None

if __name__ == "__main__":
    _synth = synthesizer.Synthesizer()
    _synth.start()

    webServer = HTTPServer((hostname, serverport), PianoServer)
    print(f"Server started http://{hostname}:{serverport}")

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    _synth.stop()
    print("Server stopped.")

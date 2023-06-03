# Copyright 2018 The dm_control Authors.
# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Components and views that render custom images into Mujoco render frame."""

import abc
import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Union

import mujoco
import numpy as np

from robopianist.viewer import renderer


class PanelLocation(enum.Enum):
    TOP_LEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT.value
    TOP_RIGHT = mujoco.mjtGridPos.mjGRID_TOPRIGHT.value
    BOTTOM_LEFT = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value
    BOTTOM_RIGHT = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT.value


class BaseViewportView(metaclass=abc.ABCMeta):
    """Base abstract view class."""

    @abc.abstractmethod
    def render(self, context, viewport, location):
        """Renders the view on screen.

        Args:
          context: MjrContext instance.
          viewport: Viewport instance.
          location: Value defined in PanelLocation enum.
        """
        pass


class ColumnTextModel(metaclass=abc.ABCMeta):
    """Data model that returns 2 columns of text."""

    @abc.abstractmethod
    def get_columns(self):
        """Returns the text to display in two columns.

        Returns:
          Returns an iterable of tuples of 2 strings. Each tuple has format
          (left_column_label, right_column_label).
        """
        pass


class ColumnTextView(BaseViewportView):
    """A text view displayed in Mujoco render window."""

    def __init__(self, model):
        """Instance initializer.

        Args:
          model: Instance of ColumnTextModel.
        """
        self._model = model

    def render(self, context, viewport, location):
        """Renders the overlay on screen.

        Args:
          context: MjrContext instance.
          viewport: Viewport instance.
          location: Value defined in PanelLocation enum.
        """
        columns = self._model.get_columns()
        if not columns:
            return

        columns = np.asarray(columns)
        left_column = "\n".join(columns[:, 0])
        right_column = "\n".join(columns[:, 1])
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            location.value,
            viewport.mujoco_rect,
            left_column,
            right_column,
            context.ptr,
        )


class MujocoDepthBuffer(renderer.Component):
    """Displays the contents of the scene's depth buffer."""

    def __init__(self):
        self._depth_buffer = np.empty((1, 1), np.float32)

    def render(self, context, viewport):
        """Renders the overlay on screen.

        Args:
          context: MjrContext instance.
          viewport: MJRRECT instance.
        """
        width_adjustment = viewport.width % 4
        rect_shape = (viewport.width - width_adjustment, viewport.height)

        if self._depth_buffer is None or self._depth_buffer.shape != rect_shape:
            self._depth_buffer = np.empty((viewport.width, viewport.height), np.float32)

        mujoco.mjr_readPixels(
            None, self._depth_buffer, viewport.mujoco_rect, context.ptr
        )

        # Subsample by 4, convert to RGB, and cast to unsigned bytes.
        depth_rgb = np.repeat(self._depth_buffer[::4, ::4, None] * 255, 3, -1).astype(
            np.uint8
        )

        pos = mujoco.MjrRect(
            int(3 * viewport.width / 4) + width_adjustment,
            0,
            int(viewport.width / 4),
            int(viewport.height / 4),
        )
        mujoco.mjr_drawPixels(depth_rgb.flatten(), None, pos, context.ptr)


class ViewportLayout(renderer.Component):
    """Layout manager for the render viewport.

    Allows to create a viewport layout by injecting renderer component even in
    absence of a renderer, and then easily reattach it between renderers.
    """

    def __init__(self):
        """Instance initializer."""
        self._views = {}

    def __len__(self):
        return len(self._views)

    def __contains__(self, key):
        value = self._views.get(key, None)
        return value is not None

    def add(self, view, location: Optional[PanelLocation]) -> None:
        """Adds a new view.

        Args:
          view: renderer.BaseViewportView instance.
          location: Value defined in PanelLocation enum, location of the view in the
            viewport.
        """
        if not isinstance(view, BaseViewportView):
            raise TypeError(
                "View added to this layout needs to implement BaseViewportView."
            )
        self._views[view] = location

    def remove(self, view) -> None:
        """Removes a view.

        Args:
          view: renderer.BaseViewportView instance.
        """
        self._views.pop(view, None)

    def clear(self):
        """Removes all attached components."""
        self._views = {}

    def render(self, context, viewport):
        """Renders the overlay on screen.

        Args:
          context: MjrContext instance.
          viewport: MJRRECT instance.
        """
        for view, location in self._views.items():
            view.render(context, viewport, location)


def panel_location_to_mujoco_rect(location: PanelLocation, viewport) -> mujoco.MjrRect:
    if location == PanelLocation.TOP_LEFT:
        pos = mujoco.MjrRect(
            0,
            int(3 * viewport.height / 4),
            int(viewport.width / 4),
            int(viewport.height / 4),
        )
    elif location == PanelLocation.TOP_RIGHT:
        pos = mujoco.MjrRect(
            int(3 * viewport.width / 4),
            int(3 * viewport.height / 4),
            int(viewport.width / 4),
            int(viewport.height / 4),
        )
    elif location == PanelLocation.BOTTOM_LEFT:
        pos = mujoco.MjrRect(
            0,
            0,
            int(viewport.width / 4),
            int(viewport.height / 4),
        )
    elif location == PanelLocation.BOTTOM_RIGHT:
        pos = mujoco.MjrRect(
            int(3 * viewport.width / 4),
            0,
            int(viewport.width / 4),
            int(viewport.height / 4),
        )
    return pos


@dataclass(frozen=True)
class TimeSeries:
    """Time series data.

    Args:
        data: A deque of floats or sequences of floats. Sequences of floats get plotted
            as multiple lines on the same figure.
        linename: A list of strings, one for each line in the time series if storing
            sequences of floats. Will be used to annotate the legend.
    """

    data: Deque[Union[float, Sequence[float]]] = field(
        default_factory=lambda: deque(maxlen=mujoco.mjMAXLINE)
    )
    linename: List[str] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.data.append(value)

    def add_dict(self, data: Dict[str, float]) -> None:
        # Sort the keys to ensure consistent ordering.
        names = sorted(data.keys())
        values = [data[name] for name in names]
        for name in names:
            if name not in self.linename:
                # MuJoCo only supports 100 characters (C-style string) for line names.
                self.linename.append(name[:99])
        self.data.append(values)

    def asarray(self) -> np.ndarray:
        """Returns the time series as a numpy array of shape (samples, lines)."""
        arr = np.asarray(self.data)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    def clear(self) -> None:
        """Clears the time series data."""
        self.data.clear()
        self.linename.clear()


class MujocoFigureModel(metaclass=abc.ABCMeta):
    """Data model that returns a MuJoCo figure."""

    def __init__(self, max_x_samples: int = mujoco.mjMAXLINE) -> None:
        self._max_x_samples = max_x_samples
        self.reset()

    def configure_figure(self) -> None:
        """Configures the MuJoCo figure. Override to customize the figure."""
        pass

    @abc.abstractmethod
    def get_time_series(self) -> Optional[TimeSeries]:
        """Returns the time series to be plotted.

        Returns:
          TimeSeries instance, or None if no data is available.
        """

    def get_figure(self) -> mujoco.MjvFigure:
        """Returns the MuJoCo figure."""
        series = self.get_time_series()
        if series is not None:
            data = series.asarray()[-self._max_x_samples :]
            self._update_lines(data)
            self._update_names(series)
        return self._figure

    def _update_names(self, series: TimeSeries) -> None:
        for i, name in enumerate(series.linename):
            self._figure.linename[i] = name

    def _update_lines(self, data: np.ndarray) -> None:
        if not data.size:
            return

        # Figure out how many lines we need to plot and tell MuJoCo.
        num_samples, num_lines = data.shape
        self._figure.linepnt[:num_lines] = num_samples

        # From the MuJoCo documentation:
        # linedata[mjMAXLINE][2*mjMAXLINEPNT]; // line data (x,y)
        # So we want to iterate over the data and fill it with consecutive x,y pairs.
        # TODO(kevin): Should we keep increasing the x value based on the episode step
        # count or keep it at is?
        for sample_index, sample in enumerate(data):
            for line_index, value in enumerate(sample):
                self._figure.linedata[line_index][2 * sample_index] = sample_index
                self._figure.linedata[line_index][2 * sample_index + 1] = value

    def reset(self) -> None:
        """Clears the figure."""
        self._figure = mujoco.MjvFigure()
        self._figure.gridsize = [5, 5]
        self._figure.range[0][:] = [0, self._max_x_samples]
        self._figure.flg_extend = 1
        self.configure_figure()


class MujocoFigureView(BaseViewportView):
    """A figure view displayed in Mujoco render window."""

    def __init__(self, model: MujocoFigureModel) -> None:
        self._model = model

    def render(self, context, viewport, location):
        figure = self._model.get_figure()
        pos = panel_location_to_mujoco_rect(location, viewport)
        mujoco.mjr_figure(pos, figure, context.ptr)

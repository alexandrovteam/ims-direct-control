from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent


def coord_formatter(data_xy, coord_xyz):
    """Formatter for matplotlib that finds the nearest point on the plot (defined by data_xy)
    and shows the corresponding coord_xyz values"""

    def format_coord(x,y):
        i = np.argmin(np.sum((data_xy[:, :2] - [[x, y]]) ** 2, axis=1))
        cx, cy, cz = coord_xyz[i, :3]
        return f'Cursor: {x:#.1f}, {y:#.1f}\nNearest: #{i} x={cx:.1f}, y={cy:.1f}, z={cz:.1f}'

    return format_coord


class PreviewPlot:
    def __init__(self, data_coords, pos_coords, logs_out):
        """
        :param data_coords: The x,y coordinates that actually get plotted
        :param pos_coords: The x,y,z physical stage-position coordinates
        """
        self.fig: plt.Figure = plt.figure()
        self.fig.canvas.mpl_connect('pick_event', self._handle_pick)
        self.ax: plt.Axes = self.fig.gca()
        self.data_coords = data_coords
        self.pos_coords = pos_coords
        self.logs_out = logs_out

        self.selected_idx: Optional[int] = None
        self.selection_cursor: Optional[plt.Line2D] = None
        self.select_listeners = []

    def __del__(self):
        self.close()

    def close(self):
        plt.close(self.fig)

    def _handle_pick(self, event: PickEvent):
        # Use log output so that errors get caught and logged in the right place
        with self.logs_out:
            # event.ind may contain many unordered points, so manually find the closest point
            point = np.array([event.mouseevent.xdata, event.mouseevent.ydata])
            idx = np.argmin(np.sum((self.data_coords[:, :2] - [point]) ** 2, axis=1))

            self.set_selection(idx)

            for listener in self.select_listeners:
                listener(idx, self.data_coords[idx], self.pos_coords[idx])

    def on_select(self, callback):
        self.select_listeners.append(callback)

    def set_selection(self, idx):
        self.selected_idx = idx
        if idx is not None:
            dx, dy = self.data_coords[idx, :2]
            if self.selection_cursor is None:
                self.selection_cursor,  = self.ax.plot([dx], [dy], marker='o', color='orange')
            else:
                self.selection_cursor.set_data(np.array([dx, dy]))
        else:
            if self.selection_cursor is not None:
                self.selection_cursor.remove()
                self.selection_cursor = None


class PhysicalPreview(PreviewPlot):
    def __init__(self, pos_coords, logs_out, safety_box):
        super().__init__(pos_coords, pos_coords, logs_out)

        x, y, z = pos_coords.T
        pos_scatter = self.ax.scatter(x, y, c=z, s=1, picker=20)

        if safety_box:
            self.ax.plot(
                [safety_box[0][0], safety_box[0][0], safety_box[1][0], safety_box[1][0], safety_box[0][0]],
                [safety_box[1][1], safety_box[0][1], safety_box[0][1], safety_box[1][1], safety_box[1][1]],
                "--r"
            )
        self.ax.axis('equal')
        self.ax.set_title("Physical shape")
        self.ax.format_coord = coord_formatter(pos_coords, pos_coords)

        self.fig.colorbar(pos_scatter)


class ImzMLCoordPreview(PreviewPlot):
    def __init__(self, imzml_coords, pos_coords, logs_out, annotate=True):
        super().__init__(imzml_coords, pos_coords, logs_out)
        dx, dy = imzml_coords[:, :2].T
        self.ax.plot(dx, dy)
        self.ax.scatter(dx, dy, s=3, picker=20)

        # Mark start and stop
        if annotate:
            self.ax.scatter(dx[0], dy[0], c=".2", marker="x")
            self.ax.scatter(dx[-1], dy[-1], c=".2", marker="x")
            self.ax.annotate("START", (dx[0], dy[0]), c=".2", xytext=(-2, 2), textcoords="offset pixels", va="bottom", ha="right")
            self.ax.annotate("STOP", (dx[-1], dy[-1]), c=".2", xytext=(3, -3), textcoords="offset pixels", va="top", ha="left")

        self.ax.axis('equal')
        self.ax.set_title("Output coordinates")
        self.ax.invert_yaxis()
        self.ax.format_coord = coord_formatter(imzml_coords, pos_coords)

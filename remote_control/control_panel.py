import time
from threading import Thread
from traceback import print_exc
from typing import Optional

import numpy as np
from IPython.display import display, clear_output
import ipywidgets as widgets
from skimage import measure, segmentation

import remote_control.control as rc
from remote_control.acquisition import Acquisition
from remote_control.utils import check_image_bounds

_current_instance: Optional['ControlPanel'] = None


def close_control_panel():
    global _current_instance
    if _current_instance:
        inst = _current_instance
        _current_instance = None
        inst.close()
        inst.update_thread.join()


def _format_position(position):
    if position is None:
        return ''
    try:
        return '(' + ', '.join([f'{v:.1f}' for v in position]) + ')'
    except:
        print_exc()
        return 'Error'



class ControlPanel:
    def __init__(self, acq: Acquisition, logs_out: widgets.Output, dummy=False):
        global _current_instance
        self.acq = acq
        self.dummy = dummy

        self.select_listeners = []

        close_control_panel()
        _current_instance = self
        self.panel_out = widgets.Output()
        self.logs_out = logs_out

        self.connected_text = widgets.Text(description='Status:', disabled=True)
        self.connect_btn = widgets.Button(description='Connect')
        self.connect_btn.on_click(self._connect)
        self.disconnect_btn = widgets.Button(description='Disconnect')
        self.disconnect_btn.on_click(self._disconnect)

        self.position_text = widgets.Text(description='Position:', disabled=True)
        self.update_pos_btn = widgets.Button(description='Update')
        self.update_pos_btn.on_click(self._update_pos)
        self.autofocus_btn = widgets.Button(description='Autofocus')
        self.autofocus_btn.on_click(self._autofocus)

        self._sel_idx = None
        self._sel_pos = None
        self.sel_pos_text = widgets.Text(description='Selected:', disabled=True)
        self.go_btn = widgets.Button(description='Go')
        self.go_btn.on_click(self._go_to_selection)
        self.trace_btn = widgets.Button(description='Preview well corners')
        self.trace_btn.on_click(self._trace_well)

        self.light_slider = widgets.IntSlider(value=50, min=0, max=100, description='Light:')
        self.light_slider.observe(self._set_light, names='value')
        self.preview_btn = widgets.Button(description='Preview full path')
        self.preview_btn.on_click(self._preview_path)

        self.targets_text = widgets.Text(description='# Pixels:', disabled=True)

        self.fly_text = widgets.Text(description='Fly height (Z):', disabled=True)

        self.update()

        layout = [
            [self.connected_text, widgets.HBox([self.connect_btn, self.disconnect_btn])],
            [self.position_text, widgets.HBox([self.update_pos_btn, self.autofocus_btn])],
            [self.sel_pos_text, widgets.HBox([self.go_btn, self.trace_btn])],
            [self.light_slider, widgets.HBox([self.preview_btn])],
            [self.targets_text],
            [self.fly_text],
        ]
        grid = widgets.GridspecLayout(len(layout), 3)
        for row, cells in enumerate(layout):
            for col, cell in enumerate(cells):
                grid[row, col] = cell

        with self.panel_out:
            display(grid)

        self.update_thread = Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()


    def on_select(self, callback):
        self.select_listeners.append(callback)

    def _select(self, idx):
        xys = np.asarray([t[0] for t in self.acq.targets])[:, :2]
        pos = np.asarray([t[1] for t in self.acq.targets])
        for listener in self.select_listeners:
            listener(idx, xys[idx], pos[idx])

    def update(self):
        if _current_instance is not self:
            return

        if self.dummy:
            connected = True
            self.connected_text.value = 'Dummy mode'
        else:
            connected = rc.telnet is not None
            self.connected_text.value = 'Connected' if connected else 'Disconnected'
        self.disconnect_btn.disabled = not connected

        self.position_text.value = _format_position(rc.last_position) or 'Unknown'
        self.update_pos_btn.disabled = not connected

        self.sel_pos_text.value = _format_position(self._sel_pos)
        self.go_btn.disabled = not connected or self._sel_pos is None
        self.trace_btn.disabled = not connected or self._sel_pos is None

        self.light_slider.disabled = not connected
        self.preview_btn.disabled = not connected or self._sel_pos is None

        self.targets_text.value = str(len(self.acq.targets))

        self.fly_text.value = str(rc.long_move_z)

    def close(self):
        with self.panel_out:
            clear_output()

    def set_selected_position(self, idx, pos_coord):
        with self.logs_out:
            self._sel_idx = idx
            self._sel_pos = pos_coord
            self.update()

    def _update_loop(self):
        while _current_instance is self:
            self.update()
            time.sleep(0.5)

    def _connect(self, b):
        with self.logs_out:
            clear_output()
            rc.initialise_and_login(self.acq.config)
            rc.set_light(self.light_slider.value)
            self.update()

    def _disconnect(self, b):
        with self.logs_out:
            clear_output()
            rc.close(True)
            self.update()

    def _update_pos(self, b):
        with self.logs_out:
            clear_output()
            rc.get_position(autofocus=False, reset_light_to=None)
            self.update()

    def _autofocus(self, b):
        with self.logs_out:
            clear_output()
            rc.get_position(autofocus=True, reset_light_to=self.light_slider.value)
            self.update()

    def _set_light(self, change):
        with self.logs_out:
            clear_output()
            rc.set_light(self.light_slider.value)

    def _go_to_selection(self, b):
        with self.logs_out:
            clear_output()
            self._go_to_position(self._sel_pos)

    def _go_to_position(self, position):
        check_image_bounds([position], self.acq.image_bounds)
        rc.acquirePixel(position, dummy=self.dummy, measure=False)

    def _trace_well(self, b):
        with self.logs_out:
            clear_output()
            # Find contiguous regions the current point is in
            idx_map, labels, xys, pos = self._get_map_and_regions()
            current_region_label = labels[tuple(xys[self._sel_idx])]
            corner_idxs = self._get_region_corners(labels == current_region_label, idx_map, pos)
            for i, corner_idx in enumerate(corner_idxs):
                if i != 0:
                    time.sleep(0.2)  # Pause after each corner

                self._select(corner_idx)
                self._go_to_position(pos[corner_idx])

    def _preview_path(self, b):
        with self.logs_out:
            clear_output()
            idx_map, labels, xys, pos = self._get_map_and_regions()
            # Find the order of regions (note that some adjustment is needed to ignore the background label 0)
            region_min_idxs = [np.min(idx_map[labels == i]) for i in np.unique(labels) if i != 0]
            label_order = np.argsort(region_min_idxs) + 1

            for i, label in enumerate(label_order):
                if i != 0:
                    time.sleep(0.5)  # Pause between wells

                corner_idxs = self._get_region_corners(labels == label, idx_map, pos)
                for i, corner_idx in enumerate(corner_idxs):
                    self._select(corner_idx)
                    self._go_to_position(pos[corner_idx])
                    if i == 0:
                        time.sleep(0.2)  # Pause at first corner


    def _get_map_and_regions(self):
        """
        Converts the ImzML acquisition coordinates to a 2D map (shifting the top-left to 0,0)
        and finds connected regions
        :return:
            idx_map: a 2D array where each cell is the index of acquisition target (or -1 for skipped)
            labels: a 2D array where 0 is skipped, and 1, 2, 3, etc. are the contiguous regions
            xys: A numpy array of the ImzML coordinates (shifted to start at 0,0 to match idx_map/labels)
            pos: A numpy array of the target physical positions
        """
        xys = np.asarray([t[0] for t in self.acq.targets])[:, :2].astype('i')
        pos = np.asarray([t[1] for t in self.acq.targets])
        xys -= np.min(xys, axis=0)
        idx_map = np.full(shape=np.max(xys, axis=0) + 1, fill_value=-1, dtype='i')
        for i, coord in enumerate(xys):
            idx_map[tuple(coord)] = i
        labels = measure.label(idx_map != -1)
        return idx_map, labels, xys, pos

    def _get_region_corners(self, region_mask, idx_map, pos):
        """
        Finds physical positions for the left and right ends of the top and bottom rows for the given region mask
        :param region_mask: A 2D boolean mask where the region of interest is set to True
        :param idx_map: From _get_map_and_regions
        :param pos: From _get_map_and_regions
        :return: List of indexes of the corner points in TL, TR, BL, BR order
        """
        filled_rows = np.flatnonzero(np.any(region_mask, axis=0))
        top_row, bottom_row = filled_rows[0], filled_rows[-1]
        top_row_cells = np.flatnonzero(region_mask[:, top_row])
        bottom_row_cells = np.flatnonzero(region_mask[:, bottom_row])

        # Find the first and last item each on the top and bottom filled row
        top_left_idx = idx_map[top_row_cells[0], top_row]
        top_right_idx = idx_map[top_row_cells[-1], top_row]
        bottom_left_idx = idx_map[bottom_row_cells[0], bottom_row]
        bottom_right_idx = idx_map[bottom_row_cells[-1], bottom_row]

        return [top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx]








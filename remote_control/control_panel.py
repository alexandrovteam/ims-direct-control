import time
from threading import Thread
from typing import Optional

from IPython.display import display, clear_output
import ipywidgets as widgets

import remote_control.control as rc
from remote_control.acquisition import Acquisition

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
    return '(' + ', '.join([f'{v:.1f}' for v in position]) + ')'


class ControlPanel:
    def __init__(self, acq: Acquisition, logs_out: widgets.Output):
        global _current_instance
        self.acq = acq

        close_control_panel()
        _current_instance = self
        self.panel_out = widgets.Output()
        self.logs_out = logs_out

        self.connected_text = widgets.Text(description='Status:', disabled=True)
        self.connect_btn = widgets.Button(description='Connect')
        self.connect_btn.on_click(self._connect)
        self.disconnect_btn = widgets.Button(description='Disconnect')
        self.connect_btn.on_click(self._disconnect)

        self.position_text = widgets.Text(description='Position:', disabled=True)
        self.update_pos_btn = widgets.Button(description='Update')
        self.update_pos_btn.on_click(self._update_pos)
        self.autofocus_btn = widgets.Button(description='Autofocus')
        self.autofocus_btn.on_click(self._autofocus)

        self._sel_pos = None
        self.sel_pos_text = widgets.Text(description='Selected:', disabled=True)
        self.go_btn = widgets.Button(description='Go')
        self.go_btn.on_click(self._go_to_selection)

        self.light_slider = widgets.IntSlider(value=50, min=0, max=100, description='Light:')
        self.light_slider.observe(self._set_light, names='value')

        self.targets_text = widgets.Text(description='# Pixels:', disabled=True)

        self.fly_text = widgets.Text(description='Fly height (Z):', disabled=True)

        self.update()

        layout = [
            [self.connected_text, widgets.HBox([self.connect_btn, self.disconnect_btn])],
            [self.position_text, widgets.HBox([self.update_pos_btn, self.autofocus_btn])],
            [self.sel_pos_text, self.go_btn],
            [self.light_slider],
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

    def update(self):
        if _current_instance is not self:
            return

        connected = rc.telnet is not None
        self.connected_text.value = 'Connected' if connected else 'Disconnected'
        self.disconnect_btn.disabled = not connected

        self.position_text.value = _format_position(rc.last_position) or 'Unknown'
        self.update_pos_btn.disabled = not connected

        self.sel_pos_text.value = _format_position(self._sel_pos)
        self.go_btn.disabled = not connected or self._sel_pos is None

        self.light_slider.disabled = not connected

        self.targets_text.value = str(len(self.acq.targets))

        self.fly_text.value = str(rc.long_move_z)

    def close(self):
        with self.panel_out:
            clear_output()

    def set_selected_position(self, pos_coord):
        with self.logs_out:
            clear_output()
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
            rc.acquirePixel(self._sel_pos, dummy=True, measure=False)


import json
import pandas as pd
import os
import numpy as np
import datetime
import json

from remote_control.utils import acquire

class Aquisition():
    def __init__(self, config_fn, datadir, log_fname=""):
        self.parse_config(config_fn)
        self.log_fname = log_fname
        self.datadir = datadir

    def parse_config(self, config_fn):
        self.config = json.load(open(config_fn))

    def acquire(self, dummy=True, image_bounds=None):
        print("Acquiring {}".format(self.dataset_name))
        xys = np.asarray([t[0] for t in self.targets])
        pos = np.asarray([t[1] for t in self.targets])
        acquire(self.config, self.log_fname, xys, pos, image_bounds, dummy, self.coords_fname)


class RectangularAquisition(Aquisition):
    def __init__(self, name, imorigin, dim_x, dim_y, pixelsize_x, pixelsize_y,  *args, **kwargs):
        self.imorigin = imorigin
        self.imagedims = [dim_x, dim_y]
        self.pixelsize = [float(pixelsize_x), float(pixelsize_y)]
        self.generate_targets()
        self.name = name
        super().__init__( *args, **kwargs)

    def generate_targets(self):
        self.targets = []
        for y in range(self.imagedims[1]):
            for x in range(self.imagedims[0]):
                self.targets.append(
                    ([x,y],
                     [-x*self.pixelsize[0]+self.imorigin[0], y*self.pixelsize[1]+self.imorigin[1], self.imorigin[2]])
                )

    @property
    def dataset_name(self):
        return "{}_{}_{}".format(self.name, self.imagedims, self.pixelsize).replace(",", "_")


    @property
    def coords_fname(self):
        return self.dataset_name + datetime.datetime.now().strftime("%Y%m%d-%hh%mm%ss") + '.positions.json'




class MaldiPlateAquisition(Aquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)





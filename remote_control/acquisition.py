import json
import pandas as pd
import os
import numpy as np
import datetime
import json

from remote_control.utils import acquire

SLIDES = {
    "sssssss",
        "spot30":
           { "spot_spacing": (6, 5, 1), #h,v (mm)
            "spot_size":(2, 2, 1), #h, v (mm)
            "grid_size":(3, 10), # h, v
           },
        "spot10":
           {"spot_spacing": (11.7, 9.7, 1), #h,v (mm) centre to centre distance
            "spot_size": (6.7, 6.7, 1), #h, v (mm)
            "grid_size": (2,5), # h, v
           }
         }

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




def get_plate_info(name):
    return [SLIDES[name][val] for val in ["spot_spacing", "spot_size", "grid_size"]]

def rms(v):
    return np.sqrt(np.square(v).sum())

pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:, :-1]

class WellPlateGridAquisition(Aquisition):
    def __init__(self, plate_type, *args, **kwargs):
        self.plate_type = plate_type
        self.plate = SLIDES[plate_type]
        self.tform = [] #transformation matrix
        print('yo, {} {}'.format(self.plate,plate_type)) 
        super().__init__(*args, **kwargs)

    def calibrate(self, instrument_positions, wells, ref_loc='centre'):
        """
        :param instrument_positions: positions of known locations from MCP (um). This should be the centre of the well.
        :param wells: x, y index of wells used for calibration.
        :return:
        """
        def get_transform(primary, secondary):
            # Pad the data with ones, so that our transformation can do translations too
            n = primary.shape[0]
            X = pad(primary)
            Y = pad(secondary)
            # Solve the least squares problem X * A = Y
            # to find our transformation matrix A
            A, res, rank, s = np.linalg.lstsq(X, Y)
            return A

        reference_positions = self.well_coord(wells, "centre")
        self.tform  = get_transform(reference_positions, instrument_positions)
        print("RMS error:", rms(instrument_positions - self.transform(reference_positions)))

    @property
    def origin(self):
        return np.asarray([self.plate["spot_size"][0]/2., self.plate["spot_size"][1]/2., 1])

    def well_coord(self, wells, location):
        LOCATIONS= ["centre", "top_left", "top_right", "bottom_left", "bottom_right"]
        TRANSFORMS = {
            "centre":       lambda wellixs: spacing * wellixs + self.origin,
            "top_left":     lambda wellixs: spacing * wellixs,
            "top_right":    lambda wellixs: spacing * wellixs + np.asarray([0, self.origin[1]]),
            "bottom_left":  lambda wellixs: spacing * wellixs + np.asarray([self.origin[0], 0]),
            "bottom_right": lambda wellixs: spacing * wellixs + 2*self.origin
        }
        assert location in LOCATIONS, "location not in {}".format(LOCATIONS)
        spacing = np.asarray(self.plate["spot_spacing"])
        transform = TRANSFORMS[location]
        return transform(np.asarray(wells))

    def transform(self, vect):
            return unpad(np.dot(pad(vect), self.tform))

    def get_measurement_bounds(self, wells_to_acquire):
        extremes = np.min(wells_to_acquire, axis=0), np.max(wells_to_acquire, axis=0)

    def acquire_wells(self,
                      wells_to_acquire,
                      pixelsize_x=50, pixelsize_y=50,
                      safety_box=None):
        """
        :param wells_to_acquire:
        :param pixelsize_x:
        :param pixelsize_y:
        :return:
        """
        measurement_bounds = self.get_measurement_bounds(wells_to_acquire)
        grid_function = self.get_grid_function()


        pass






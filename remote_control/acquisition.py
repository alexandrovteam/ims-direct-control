import json
import pandas as pd
import os
import numpy as np
import datetime
import json
from scipy import interpolate
from skimage import measure
from remote_control.utils import acquire

SLIDES = {
        "spot30":
           { "spot_spacing": (6, 5, 1), #h,v (mm)
            "spot_size":(2., 2., 1.), #h, v (mm)
            "grid_size":(3, 10), # h, v
             "shape": "circle",
           },
        "spot10":
           {"spot_spacing": (11.7, 9.7, 1), #h,v (mm) centre to centre distance
            "spot_size": (6.7, 6.7, 1), #h, v (mm)
            "grid_size": (2,5), # h, v,
            "shape": "circle",
           },
        "labtek":
           {"spot_spacing": (1.2, 1.2, 1), #h,v (mm) centre to centre distance
            "spot_size": (3., 2., 1.), #h, v (mm)
            "grid_size": (1, 4), # h, v,
            "shape": "rectangle",
           }
         }

MASK_FUNCTIONS = {
        "circle": lambda xv, yv, r, c: np.square(xv - c[0])/(np.min(r)/2)**2 + np.square(yv - c[1])/(np.min(r)/2) ** 2 < 1,
        "ellipse": lambda xv, yv, r, c: np.square(xv - c[0])/(r[0]/2)**2 + np.square(yv - c[1])/(r[1]/2) ** 2 < 1,
        "rectangle": lambda xv, yv, r, c: (xv < c[0] + r[0]/2.) & (xv > c[0] - r[0]/2.) & (yv < c[1] + r[1]/2.) & (yv > c[1] - r[1]/2.),
}

AREA_FUNCTIONS = {
    None: lambda xv, yv, r, c: True,
    "left": lambda xv, yv, r, c: (xv < c[0]),
    "right": lambda xv, yv, r, c: (xv > c[0]),
    "upper": lambda xv, yv, r, c: (yv > c[1]),
    "lower": lambda xv, yv, r, c: (yv < c[1]),
    "upper_left": lambda xv, yv, r, c: (xv < c[0]) & (yv > c[1]),
    "upper_right": lambda xv, yv, r, c: (xv > c[0]) & (yv > c[1]),
    "lower_left": lambda xv, yv, r, c: (xv < c[0]) & (yv < c[1]),
    "lower_right": lambda xv, yv, r, c: (xv > c[0]) & (yv < c[1]),
}


def get_plate_info(name):
    return [SLIDES[name][val] for val in ["spot_spacing", "spot_size", "grid_size"]]

def rms(v):
    return np.sqrt(np.square(v).sum())

pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:, :-1]

class Aquisition():
    def __init__(self, config_fn, datadir, log_fname=""):
        self.parse_config(config_fn)
        self.log_fname = log_fname
        self.datadir = datadir
        self.targets = []

    def parse_config(self, config_fn):
        self.config = json.load(open(config_fn))

    def coords_fname(self, dataset_name):
        raise NotImplementedError

    def acquire(self, dataset_name, dummy=True, image_bounds=None):
        print("Acquiring {}".format(dataset_name))
        xys = np.asarray([t[0] for t in self.targets])
        pos = np.asarray([t[1] for t in self.targets])
        acquire(self.config, self.log_fname, xys, pos, image_bounds, dummy, self.coords_fname(dataset_name))

    def mask_function(self, mask_function_name):
        return MASK_FUNCTIONS[mask_function_name]

    def area_function(self, area_function_name):
        return AREA_FUNCTIONS[area_function_name]

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

    def coords_fname(self, dataset_name):
        return dataset_name + datetime.datetime.now().strftime("%Y%m%d-%hh%mm%ss") + '.positions.json'


class WellPlateGridAquisition(Aquisition):
    def __init__(self, plate_type, *args, **kwargs):
        self.plate_type = plate_type
        self.plate = SLIDES[plate_type]
        self.tform = [] #transformation matrix
        super().__init__(*args, **kwargs)

    def coords_fname(self, dataset_name):
        return dataset_name + 'positions.json'

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
            A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
            return A
        instrument_positions, wells = map(lambda x: np.asarray(x), [instrument_positions, wells])
        reference_positions = self.well_coord(wells, ref_loc)
        self.tform  = get_transform(reference_positions, instrument_positions)
        print("RMS error:", rms(instrument_positions - self.transform(reference_positions)))

    @property
    def origin(self):
        return np.asarray([self.plate["spot_size"][0]/2., self.plate["spot_size"][1]/2., 0])

    def well_coord(self, wells, location):
        LOCATIONS= ["centre", "top_left", "top_right", "bottom_left", "bottom_right"]
        TRANSFORMS = {
            "centre":       lambda wellixs: spacing * wellixs + self.origin,
            "top_left":     lambda wellixs: spacing * wellixs,
            "top_right":    lambda wellixs: spacing * wellixs + np.asarray([0, self.origin[1], 0]),
            "bottom_left":  lambda wellixs: spacing * wellixs + np.asarray([self.origin[0], 0, 0]),
            "bottom_right": lambda wellixs: spacing * wellixs + 2*self.origin
        }
        assert location in LOCATIONS, "location not in {}".format(LOCATIONS)
        spacing = np.asarray(self.plate["spot_spacing"])
        transform = TRANSFORMS[location]
        return transform(np.asarray(wells))

    def transform(self, vect):
        return unpad(np.dot(pad(vect), self.tform))

    def get_measurement_bounds(self, wells_to_acquire):
        wells_to_acquire = np.asarray(wells_to_acquire)
        mins = np.min(wells_to_acquire, axis=0)
        maxs = np.max(wells_to_acquire, axis=0)
        extremes =[
            [mins[0], mins[1], 0],
            [mins[0], maxs[1], 0],
            [maxs[0], mins[1], 0],
            [maxs[0], maxs[1], 0]
        ]
        locations = ["top_left", "top_right", "bottom_left", "bottom_right"]
        t = [self.transform(self.well_coord(e,l).reshape(1,-1))[0] for e,l in zip(extremes, locations)]
        return np.asarray(t)

    def write_imzml_coords(self, dataset_name):
        import json
        fn = self.coords_fname(dataset_name)
        coords = json.load(open(fn))
        fn2 = fn.replace(".json", "imzc.txt")
        with open(fn2, "w") as f:
            for x, y in zip(coords['index x'], coords['index y']):
                f.write("{} {}\n".format(int(x), int(y)))


    def generate_targets(self, wells_to_acquire, pixelsize_x, pixelsize_y,
                                    offset_x, offset_y,
                                    mask_function_name=None, area_function_name=None):
        if mask_function_name is None:
            mask_function_name = self.plate['shape']

        measurement_bounds = self.get_measurement_bounds(wells_to_acquire)

        x0, y0 = measurement_bounds.min(axis=0)[0:2]
        xmax, ymax = measurement_bounds.max(axis=0)[0:2]
        x = np.arange(x0, xmax, pixelsize_x)
        y = np.arange(y0, ymax, pixelsize_y)

        _z = interpolate.interp2d(measurement_bounds[:, 0], measurement_bounds[:, 1], measurement_bounds[:, 2])
        xv, yv = np.meshgrid(x, y)
        mask = np.zeros(xv.shape)
        r = [_d*1000 for _d in self.plate["spot_size"]]

        for well in wells_to_acquire:
            c = self.transform(self.well_coord(np.asarray([well[0], well[1], 0]).reshape(1, -1), 'centre'))[0]
            mask[
                self.mask_function(mask_function_name)(xv, yv, r, c)
                * self.area_function(area_function_name)(xv, yv, r, c)
            ] += 1

        mask_labels = measure.label(mask, background=0)
        self.targets = []
        for ii in range(1, np.max(mask_labels) + 1):
            _xy = list([
                (
                    ((_x - x0) / pixelsize_x, (_y - y0) / pixelsize_y),  # pixel index (x,y)
                    (_x + offset_x, _y + offset_y, _z(_x, _y))  # absolute position (x,y,z)

                )
                for _x, _y in zip(xv[mask_labels == ii].flatten(), yv[mask_labels == ii].flatten())
            ])
            self.targets.extend(_xy)

    def acquire_wells(self,
                      wells_to_acquire,
                      dataset_name,
                      dummy=True,
                      pixelsize_x=50, pixelsize_y=50,
                      offset_x = 0, offset_y=0,
                      area_shape="circle",
                      area_mask = None,
                      safety_box=None):
        """
        :param wells_to_acquire: index (x,y) of wells to image
        :param dataset_name: output filename (should match .raw filename)
        :param dummy: dummy run (True, False)
        :param pixelsize_x: spatial separation in x (um)
        :param pixelsize_y: spatial separation in y (um)
        :param offset_x: (default=0) offset from 0,0 position for acquisition points in x (um)
        :param offset_y: (default=0) offset from 0,0 position for acquisition points in y (um)
        :param safety_box: (default=None) additional bounding box (um) to constrain measurements within motor travel region.
        :return:
        """
        self.generate_targets(wells_to_acquire,
                              pixelsize_x, pixelsize_y,
                              offset_x, offset_y,
                              area_shape, area_mask)
        print("total pixels: ", len(self.targets))
        if dummy:
            import matplotlib.pyplot as plt
            xys = np.asarray([t[0] for t in self.targets])
            pos = [t[1] for t in self.targets]
            print(xys)
            plt.figure()
            plt.plot([xy[0] for xy in xys], [xy[1] for xy in xys])
            plt.scatter([xy[0] for xy in xys], [xy[1] for xy in xys])
            plt.axis('equal')
            plt.show()

            plt.figure()
            plt.scatter([xy[0] for xy in pos], [xy[1] for xy in pos])
            plt.plot(
                [safety_box[0][0], safety_box[0][0], safety_box[1][0], safety_box[1][0]],
                [safety_box[1][1], safety_box[0][1], safety_box[0][1], safety_box[1][1]],
                "--r"
            )
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.show()

        self.acquire(dataset_name=dataset_name, dummy=dummy, image_bounds=safety_box)
        self.write_imzml_coords(dataset_name)
        return self.targets






from datetime import datetime
from functools import wraps
from inspect import getcallargs
from pathlib import Path

import numpy as np
import json
from scipy import interpolate
from scipy.spatial.distance import euclidean, cosine
from skimage import measure
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from remote_control.control import save_coords
from remote_control.utils import acquire, NpEncoder

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
    "circle": lambda xv, yv, r, c: np.square(xv - c[0])/((r[0]/2)**2) + np.square(yv - c[1])/((r[0]/2)** 2) <= 1,
    "ellipse": lambda xv, yv, r, c: np.square(xv - c[0])/(r[0]/2)**2 + np.square(yv - c[1])/(r[1]/2) ** 2 < 1,
    "rectangle": lambda xv, yv, r, c: (xv < c[0] + r[0]/2.) & (xv > c[0] - r[0]/2.) & (yv < c[1] + r[1]/2.) & (yv > c[1] - r[1]/2.),
}

AREA_FUNCTIONS = {
    None: lambda xv, yv, r, c, m: True,
    "left": lambda xv, yv, r, c, m: (xv < c[0] - m),
    "right": lambda xv, yv, r, c, m: (xv > c[0] + m),
    "upper": lambda xv, yv, r, c, m: (yv > c[1] + m),
    "lower": lambda xv, yv, r, c, m: (yv < c[1] - m),
    "upper_left": lambda xv, yv, r, c, m: (xv < c[0] - m) & (yv > c[1] + m),
    "upper_right": lambda xv, yv, r, c, m: (xv > c[0] + m) & (yv > c[1] + m),
    "lower_left": lambda xv, yv, r, c, m: (xv < c[0] - m) & (yv < c[1] - m),
    "lower_right": lambda xv, yv, r, c, m: (xv > c[0] + m) & (yv < c[1] - m),
}


def get_plate_info(name):
    return [SLIDES[name][val] for val in ["spot_spacing", "spot_size", "grid_size"]]

def rms(v):
    return np.sqrt(np.square(v).sum())

def unit_vector(v):
    return v / np.linalg.norm(v)

pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:, :-1]


def coord_formatter(data_x, data_y, data_z):
    data_x, data_y, data_z = map(np.array, [data_x, data_y, data_z])

    def format_coord(x,y):
        i = np.argmin((data_x - x) ** 2 + (data_y - y) ** 2)
        return 'Cursor: x=%#.5g, y=%#.5g\nNearest: #%i x=%#.5g, y=%#.5g, z=%#.5g' % (x, y, i, data_x[i], data_y[i], data_z[i])

    return format_coord


def grid_deviation(coords):
    """Models a linear transform based on the 4 supplied coords and returns the maximum error for each axis"""
    base_coords = [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    coords = np.array(list(coords))
    tform = np.linalg.lstsq(base_coords, coords, rcond=None)[0]
    result_coords = np.array([np.dot(base_coord, tform) for base_coord in base_coords])
    error = np.max(np.abs(result_coords - coords), axis=0)
    return error


def grid_skew(coords):
    ll, lh, hl, hh = coords[:4]
    x_vector = unit_vector(ll - lh + hl - hh)
    y_vector = unit_vector(ll - hl + lh - hh)
    grid_skew = np.dot(x_vector, y_vector)
    return np.rad2deg(np.arcsin(grid_skew))


def _record_args(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, '_recorded_args', None) is None:
            self._recorded_args = {'__class__': self.__class__.__name__}
        all_args = getcallargs(func, self, *args, **kwargs)
        all_args.pop('self', None)

        self._recorded_args[func.__name__] = all_args
        return func(self, *args, **kwargs)
    return wrapper

class Acquisition():
    def __init__(self, config_fn, datadir=None):
        self.config = json.load(open(config_fn))
        self.datadir = datadir
        self.targets = []

        self.subpattern_coords = [(0, 0)] # Physical X/Y offsets (in µm)
        self.subpattern_pixels = [(0, 0)] # Pixel-space X/Y offsets

    def coords_fname(self, dataset_name):
        return dataset_name + 'positions.json'

    def write_imzml_coords(self, dataset_name):
        import json
        fn = self.coords_fname(dataset_name)
        coords = json.load(open(fn))
        fn2 = fn.replace(".json", "imzc.txt")
        with open(fn2, "w+") as f:
            for x, y in zip(coords['index x'], coords['index y']):
                f.write("{} {}\n".format(int(x), int(y)))

    def write_imzc_coords_file(self, filename):
        xys = np.asarray([t[0] for t in self.targets])
        with open(filename, "w") as f:
            for x, y in xys[:, :2]:
                f.write("{} {}\n".format(int(x), int(y)))

    def write_json_coords_file(self, filename):
        xys = np.asarray([t[0] for t in self.targets])
        pos = np.asarray([t[1] for t in self.targets])
        save_coords(filename, xys, pos, [], [])

    @_record_args
    def set_image_bounds(self, image_bounds):
        self.image_bounds = image_bounds

    @_record_args
    def apply_subpattern(self, subpattern_pixels, subpattern_coords, size_x, size_y):
        self.subpattern_pixels = subpattern_pixels
        self.subpattern_coords = subpattern_coords

        self.targets = [
            ((px * size_x + pox, py * size_y + poy), (cx + cox, cy + coy, cz))
            for (px, py), (cx, cy, cz) in self.targets
            for (pox, poy), (cox, coy) in zip(self.subpattern_pixels, self.subpattern_coords)
        ]

    @_record_args
    def apply_spiral_subpattern(self, spacing_x, spacing_y):
        subpattern_pixels = [
            (1,1),
            (1,0),
            (2,0),
            (2,1),
            (2,2),
            (1,2),
            (0,2),
            (0,1),
            (0,0)
        ]
        subpattern_coords = (np.array(subpattern_pixels) * [[spacing_x, spacing_y]]).tolist()
        psx, psy = np.max(subpattern_pixels, axis=0) + 1 # size of pixel grid

        self.apply_subpattern(subpattern_pixels, subpattern_coords, psx, psy)

    def _save_recorded_args(self, suffix=''):
        if self.config.get('saved_parameters') and getattr(self, '_recorded_args', None) is not None:
            base_path = Path(self.config.get('saved_parameters'))
            base_path.mkdir(parents=True, exist_ok=True)
            f = base_path / f'{datetime.now().isoformat().replace(":","_")}{suffix}.json'
            json.dump(self._recorded_args, f.open('w'), indent=2, cls=NpEncoder)

    @_record_args
    def acquire(self, dataset_name, dummy=True, measure=True):
        """ DEPRECATED - use generate_targets and acquire separately instead
        :param dataset_name: output filename (should match .raw filename)
        :param dummy: dummy run (True, False)
        :param measure: True to measure, False to only send goto commands (True, False)
        :return:
        """
        print("Acquiring {} ({} pixels)".format(dataset_name, len(self.targets)))
        xys = np.asarray([t[0] for t in self.targets])
        pos = np.asarray([t[1] for t in self.targets])
        self._recorded_args['raw_xys'] = xys.tolist()
        self._recorded_args['raw_pos'] = pos.tolist()
        self._save_recorded_args('_dummy' if dummy else '_moveonly' if not measure else '_real')
        acquire(
            self.config,
            xys,
            pos,
            self.image_bounds,
            dummy,
            self.coords_fname(dataset_name),
            measure=measure
        )
        if not dummy:
            self.write_imzml_coords(dataset_name)

    def mask_function(self, mask_function_name):
        return MASK_FUNCTIONS[mask_function_name]

    def area_function(self, area_function_name):
        return AREA_FUNCTIONS[area_function_name]

    def apply_image_mask(self, filename, threshold=0.5):
        import matplotlib.pyplot as plt
        img = np.atleast_3d(plt.imread(filename))
        mask = np.mean(img, axis=2) > threshold
        self.targets = [([cx, cy], pos) for ((cx, cy), pos) in self.targets
                        if cx < mask.shape[0] and cy < mask.shape[1] and mask[cx, cy]]
        print(f'Number of pixels after mask: {len(self.targets)}')

    def plot_targets(self, annotate=False, show=True):
        """ Plot output data coordinates and physical coordinates.
        :param annotate: bool, whether to annotate start and stop.
        :param show: bool, whether to show the plots, if False return it instead.
        :return: a tuple of two plt.Figure objects containing the plots if show == False.
        """
        import matplotlib.pyplot as plt
        safety_box = self.image_bounds
        xys = np.asarray([t[0] for t in self.targets])
        pos = [t[1] for t in self.targets]

        print("total pixels: ", len(self.targets))

        path_fig = plt.figure()
        plt.plot([xy[0] for xy in xys], [xy[1] for xy in xys])
        plt.scatter([xy[0] for xy in xys], [xy[1] for xy in xys], s=3)

        # Mark start and stop
        if annotate:
            plt.scatter(*xys[0], c=".2", marker="x")
            plt.scatter(*xys[-1], c=".2", marker="x")
            plt.annotate("START", xys[0], c=".2", xytext = (-2, 2), textcoords="offset pixels", va="bottom", ha="right")
            plt.annotate("STOP", xys[-1], c=".2", xytext = (3, -3), textcoords="offset pixels", va="top", ha="left")

        plt.axis('equal')
        plt.title("Output coordinates")
        plt.gca().invert_yaxis()
        plt.gca().format_coord = coord_formatter([xy[0] for xy in xys], [xy[1] for xy in xys], [0 for xy in xys])
        if show:
            plt.show()

        pos_fig = plt.figure()
        plt.scatter([xy[0] for xy in pos], [xy[1] for xy in pos], c=[xy[2] for xy in pos], s=1)
        plt.plot(
            [safety_box[0][0], safety_box[0][0], safety_box[1][0], safety_box[1][0], safety_box[0][0]],
            [safety_box[1][1], safety_box[0][1], safety_box[0][1], safety_box[1][1], safety_box[1][1]],
            "--r"
        )
        plt.axis('equal')
        plt.title("Physical shape")
        plt.gca().format_coord = coord_formatter([xy[0] for xy in pos], [xy[1] for xy in pos], [xy[2] for xy in pos])
        plt.colorbar()
        
        if show:
            plt.show()
        else:
            return path_fig, pos_fig



class RectangularAquisition(Acquisition):
    @_record_args
    def generate_targets(self, calibration_positions, target_positions,
                         x_pitch=None, y_pitch=None,
                         x_size=None, y_size=None,
                         interpolate_xy=False):
        """
        :param calibration_positions: Coordinates used for calculating the Z axis.
        :param target_positions: Coordinates of the 4 corners to sample
        :param x_pitch: Distance between pixels in the X axis (calculated from x_size if needed)
        :param y_pitch: Distance between pixels in the Y axis (calculated from y_size if needed)
        :param x_size: Number of pixels in the X axis (calculated from x_pitch if needed)
        :param y_size: Number of pixels in the Y axis (calculated from y_pitch if needed)
        :param interpolate_xy: False to use a linear transform for calculating X/Y, which ensures the shape is at least
                               a parallelogram so that the x/y pitch is consistent.
                               True to use interpolation, which allows the shape to be trapezoidal, which ensures that
                               target_positions are hit exactly, but can lead to uneven pitches

        """

        # Normalize inputs
        calibration_positions = np.array(calibration_positions)
        target_positions = np.array(target_positions)[:, :2]

        if x_pitch is not None and x_size is None:
            x_size = int(euclidean(target_positions[0], target_positions[1]) / x_pitch)
        elif x_pitch is None and x_size is not None:
            x_pitch = euclidean(target_positions[0], target_positions[1]) / x_size
        else:
            raise ValueError("either x_pitch or x_size must be specified, but not both")

        if y_pitch is not None and y_size is None:
            y_size = int(euclidean(target_positions[0], target_positions[2]) / y_pitch)
        elif y_pitch is None and y_size is not None:
            y_pitch = euclidean(target_positions[0], target_positions[2]) / y_size
        else:
            raise ValueError("either y_pitch or y_size must be specified, but not both")

        # Calculate the coordinate frames and print debug info
        corner_coords = np.array([(0, 0, 1), (x_size, 0, 1), (0, y_size, 1), (x_size, y_size, 1)])
        print(f"Output size: {x_size} x {y_size} pixels (= {x_size*y_size} total pixels)")
        print(f"Output grid pitch: {x_pitch:#.5g} x {y_pitch:#.5g}")

        xy_to_z = interpolate.interp2d(calibration_positions[:, 0], calibration_positions[:, 1], calibration_positions[:, 2])
        error_x, error_y, error_z = grid_deviation([(x, y, *xy_to_z(x, y)) for x, y in target_positions])

        print(f"Maximum error due to grid irregularity: x±{error_x:#.2f}, y±{error_y:#.2f}, z±{error_z:#.2f}")
        print(f"Grid skew: {grid_skew(target_positions):#.1f}°")

        if interpolate_xy:
            coord_to_x = interpolate.interp2d(corner_coords[:, 0], corner_coords[:, 1], target_positions[:, 0])
            coord_to_y = interpolate.interp2d(corner_coords[:, 0], corner_coords[:, 1], target_positions[:, 1])
            coord_to_xy = lambda cx, cy: (coord_to_x(cx, cy).item(0), coord_to_y(cx, cy).item(0))
        else:
            coord_to_xy_matrix = np.linalg.lstsq(corner_coords, target_positions, rcond=None)[0]
            coord_to_xy = lambda cx, cy: tuple(np.dot((cx, cy, 1), coord_to_xy_matrix).tolist())

        # Write all coordinates to self.targets
        self.targets = []
        for cy in range(y_size):
            for cx in range(x_size):
                x_pos, y_pos = coord_to_xy(cx, cy)
                z_pos = xy_to_z(x_pos, y_pos).item(0)
                self.targets.append(([cx, cy], [x_pos, y_pos, z_pos]))


class WellPlateGridAquisition(Acquisition):
    @_record_args
    def __init__(self, plate_type, *args, **kwargs):
        if isinstance(plate_type, dict):
            self.plate_type = plate_type['name']
            self.plate=plate_type
        else:
            self.plate_type = plate_type
            self.plate = SLIDES[plate_type]
        self.tform = [] #transformation matrix
        super().__init__(*args, **kwargs)

    @_record_args
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
        reference_positions = self._well_coord(wells, ref_loc)
        self.tform  = get_transform(reference_positions, instrument_positions)
        print("RMS error:", rms(instrument_positions - self._transform(reference_positions)))

    @property
    def _origin(self):
        return np.asarray([self.plate["spot_size"][0]/2., self.plate["spot_size"][1]/2., 0])

    def _well_coord(self, wells, location):
        LOCATIONS= ["centre", "top_left", "top_right", "bottom_left", "bottom_right"]
        TRANSFORMS = {
            "centre":       lambda wellixs: spacing * wellixs + self._origin,
            "top_left":     lambda wellixs: spacing * wellixs,
            "top_right":    lambda wellixs: spacing * wellixs + np.asarray([2 * self._origin[0], 0, 0]),
            "bottom_left":  lambda wellixs: spacing * wellixs + np.asarray([0, 2 * self._origin[1], 0]),
            "bottom_right": lambda wellixs: spacing * wellixs + 2*self._origin
        }
        assert location in LOCATIONS, "location not in {}".format(LOCATIONS)
        spacing = np.asarray(self.plate["spot_spacing"])
        transform = TRANSFORMS[location]
        return transform(np.asarray(wells))

    def _transform(self, vect):
        return unpad(np.dot(pad(vect), self.tform))

    def _get_measurement_bounds(self, wells_to_acquire):
        wells_to_acquire = np.asarray(wells_to_acquire)
        mins = np.min(wells_to_acquire, axis=0)
        maxs = np.max(wells_to_acquire, axis=0)
        extremes =[
            [mins[0], mins[1], 0],
            [maxs[0], mins[1], 0],
            [mins[0], maxs[1], 0],
            [maxs[0], maxs[1], 0]
        ]
        locations = ["top_left", "top_right", "bottom_left", "bottom_right"]
        t = [self._transform(self._well_coord(e, l).reshape(1, -1))[0] for e, l in zip(extremes, locations)]
        return np.asarray(t)

    @_record_args
    def generate_targets(self, wells_to_acquire, pixelsize_x, pixelsize_y,
                         offset_x, offset_y,
                         mask_function_name=None, area_function_name=None,
                         area_function_margin=0, shared_grid=False):
        """
        :param wells_to_acquire: index (x,y) of wells to image
        :param pixelsize_x: spatial separation in x (um)
        :param pixelsize_y: spatial separation in y (um)
        :param offset_x: (default=0) offset from 0,0 position for acquisition points in x (um)
        :param offset_y: (default=0) offset from 0,0 position for acquisition points in y (um)
        :param mask_function_name: None, 'circle', 'ellipse', 'rectangle'
        :param area_function_name: None, 'left', 'upper', 'upper_left', etc.
        :param area_function_margin: distance (um) between opposing areas defined by area function
        :param shared_grid: if True, one big grid is used for the whole acquisition,
                            so pixels are perfectly evenly spaced, even between wells.
                            This allows the optical image to perfectly match the ablated area
                            if False, each well gets its own pixel grid. This allows a better fit
                            for the well shape, but may physically be up to 1 pixelsize away from
                            the optically registered point.
        :return:
        """

        if mask_function_name is None:
            print(self.plate)
            mask_function_name = self.plate['shape']

        def well_mask(c, xv, yv):
            r = [_d * 1000 for _d in self.plate["spot_size"]]
            if np.round(r[0] / pixelsize_x):
                # odd number of pixels wide, aim for the center of a pixel
                c[0] = (np.round(c[0] / pixelsize_x + 0.5) - 0.5) * pixelsize_x
            else:
                c[0] = np.round(c[0] / pixelsize_x) * pixelsize_x
            if np.round(r[1] / pixelsize_y):
                # odd number of pixels tall, aim for the center of a pixel
                c[1] = (np.round(c[1] / pixelsize_y + 0.5) - 0.5) * pixelsize_y
            else:
                c[1] = np.round(c[1] / pixelsize_y) * pixelsize_y

            return (
                self.mask_function(mask_function_name)(xv, yv, r, c)
                * self.area_function(area_function_name)(xv, yv, r, c, area_function_margin / 2)
            )

        if shared_grid:
            self._generate_targets_single_grid(
                wells_to_acquire, pixelsize_x, pixelsize_y,
                offset_x, offset_y, well_mask
            )
        else:
            self._generate_targets_grid_per_well(
                wells_to_acquire, pixelsize_x, pixelsize_y,
                offset_x, offset_y, well_mask
            )

    def _generate_targets_single_grid(self, wells_to_acquire, pixelsize_x, pixelsize_y,
                                      offset_x, offset_y, well_mask):

        measurement_bounds = self._get_measurement_bounds(wells_to_acquire)

        x0, y0 = measurement_bounds.min(axis=0)[0:2]
        xmax, ymax = measurement_bounds.max(axis=0)[0:2]
        x = np.arange(x0, xmax, pixelsize_x)
        y = np.arange(y0, ymax, pixelsize_y)[::-1]

        _z = interpolate.interp2d(measurement_bounds[:, 0], measurement_bounds[:, 1], measurement_bounds[:, 2])
        xv, yv = np.meshgrid(x, y)
        mask = np.zeros(xv.shape)

        for well in wells_to_acquire:
            c = self._transform(self._well_coord([[well[0], well[1], 0]], 'centre'))[0]
            mask[well_mask(c, xv, yv)] += 1

        mask_labels = measure.label(mask, background=0)
        self.targets = []
        for ii in range(1, np.max(mask_labels) + 1):
            _xy = list([
                (
                    ((_x - x0) / pixelsize_x, (ymax - _y - y0) / pixelsize_y),  # pixel index (x,y)
                    (_x + offset_x, _y + offset_y, _z(_x, _y)[0])  # absolute position (x,y,z)
                )
                for _x, _y in zip(xv[mask_labels == ii].flatten(), yv[mask_labels == ii].flatten())
            ])
            self.targets.extend(_xy)

    def _generate_targets_grid_per_well(self, wells_to_acquire, pixelsize_x, pixelsize_y,
                                        offset_x, offset_y, well_mask):

        measurement_bounds = self._get_measurement_bounds(wells_to_acquire)

        x0, y0 = measurement_bounds.min(axis=0)[0:2]
        xmax, ymax = measurement_bounds.max(axis=0)[0:2]

        _z = interpolate.interp2d(measurement_bounds[:, 0], measurement_bounds[:, 1], measurement_bounds[:, 2])
        spot_size = np.array(self.plate["spot_size"][:2]) * 1000
        dim_x, dim_y = np.int64(np.round(spot_size / [pixelsize_x, pixelsize_y]))

        def coords_for_well(well):
            well_x, well_y = self._transform(self._well_coord([[*well, 0]], 'centre'))[0][:2] - spot_size / 2
            return np.meshgrid(
                np.arange(dim_x) * pixelsize_x + well_x,
                (np.arange(dim_y) * pixelsize_y + well_y)[::-1]
            )

        template_xv, template_yv = np.meshgrid(
            np.arange(dim_x) * pixelsize_x - spot_size[0] / 2,
            (np.arange(dim_y) * pixelsize_y - spot_size[1] / 2)[::-1]
        )
        mask = well_mask(np.array([0, 0, 0]), template_xv, template_yv)

        self.targets = []
        for well in wells_to_acquire:
            xv, yv = coords_for_well(well)
            self.targets.extend([
                (
                    ((_x - x0) / pixelsize_x, (ymax - _y - y0) / pixelsize_y),  # pixel index (x,y)
                    (_x + offset_x, _y + offset_y, _z(_x, _y)[0])  # absolute position (x,y,z)
                )
                for _x, _y in zip(xv[mask].flatten(), yv[mask].flatten())
            ])


class AcquistionArea():
    """
    Class to define a rectangular acquisition area (Used by QueueAcquisition)
    """
    # @_record_args
    def __init__(self, line_start, line_end, perpendicular, step_size_x, step_size_y, name=None):
        self.ls = line_start
        self.le = line_end
        self.pp = perpendicular
        self.ssx = step_size_x
        self.ssy = step_size_y
        self.name = name

        # Compute grid size
        self.res_x = int(np.ceil(abs(self.le[0] - self.ls[0]) / self.ssx))
        self.res_y = int(np.ceil(abs(self.pp[1] - self.ls[1]) / self.ssy))
        self.n_acquisitions = self.res_x * self.res_y
        
        self.targets = None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', res_x={self.res_x}, res_y={self.res_y}, n_acquisitions={self.n_acquisitions})"


class QueueAquisition(Acquisition):
    """
    Acquistion type that allows queueing of rectangular areas for successive acquisition
    """
    @_record_args
    def __init__(self, *args, **kwargs):
        self.queue = []
        super().__init__(*args, **kwargs)

    def add_area(self, line_start, line_end, perpendicular, step_size_x, step_size_y, name=None):
        """
        Create and add an AcquisitionArea to the queue.
        """
        
        area = AcquistionArea(line_start, line_end, perpendicular, step_size_x, step_size_y, name=name)
        self.queue.append(area)
        
        return area

    def clear_areas(self):
        self.queue = []
        self.targets = []

    def plot_areas(self, plot_labtek_wells=False, show=True):
        """
        Plot acquisition areas.

        :param plot_labtek_wells: bool, whether to include a 2x4 well grid into the plot starting from top left image bound.
        :param show: bool, whether to show the plot, if False return it instead.
        :return: a plt.Figure object containing the plot if show == False
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        print("total areas:", len(self.queue))

        safety_box = self.image_bounds
        
        fig = plt.figure()
        
        plt.plot(
            [safety_box[0][0], safety_box[0][0], safety_box[1][0], safety_box[1][0], safety_box[0][0]],
            [safety_box[1][1], safety_box[0][1], safety_box[0][1], safety_box[1][1], safety_box[1][1]],
            "0.8",
            linestyle=":",
            linewidth=1
        )

        ax = fig.axes[0]

        # This is hard-coded badness. Don't!
        if plot_labtek_wells:
            origin_x, origin_y = self.image_bounds[1]
            well_w, well_h = 7000, 9000
            ctc_x, ctc_y = 10500, 12500

            for col_idx in range(2):
                for row_idx in range(4):
                    g = patches.Rectangle(
                        (
                            origin_x + (ctc_x * col_idx), 
                            origin_y - (ctc_y * row_idx)
                        ),
                        width = well_w, 
                        height = -well_h,
                        fill = False,
                        edgecolor = "0.8",
                        linewidth = 1
                    )
                    ax.add_patch(g)
                    


        for idx, area in enumerate(self.queue):
            g = patches.Rectangle(
                (area.ls[0], area.pp[1]),
                width = area.le[0]-area.ls[0], 
                height = area.ls[1]-area.pp[1],
                fill = False,
                edgecolor = "C0",
                alpha=.5,
                hatch="////"
            )
            ax.add_patch(g)

            plt.annotate(
                text=idx,
                xy=(area.ls[0] - (area.ls[0]-area.le[0])/2,
                    area.ls[1] - (area.ls[1]-area.pp[1])/2),
                ha="center",
                va="center",
                c="C0",
                fontsize=12
            )
            if area.name:
                plt.annotate(area.name, area.ls[0:2], va="bottom", ha="right")

        plt.axis('equal')

        plt.title("Acquistion areas")
        
        if show:
            plt.show()
        else:
            return fig
        
    
    @_record_args
    def generate_targets(self, meander=False):
        """
        Create the targets from queued areas.

        :param meander: whether to scan every even-numbered row in reverse instead of jumping back to the row start (default False)
        
        """
        
        origin = self.image_bounds[1] # Pixel indices will be relative to top-left
        self.targets = []
        
        for area in self.queue:

            fixpoints = np.asarray([area.ls, area.le, area.pp])

            # Fit linear model to fixpoints to create 3D plane
            lr = LinearRegression()
            plane = lr.fit(fixpoints[:, :2], fixpoints[:, 2]) # fit to LS, LE, PP

            # Create XY coordinates for targets
            target_xs, target_ys = np.meshgrid(
                np.arange(
                    *np.sort([area.ls[0], area.le[0]]),
                    area.ssx),
                np.arange(
                    *np.sort([area.ls[1], area.pp[1]])[::-1], # invert since we scan top to bottom
                    area.ssy * -1
                )
            )

            if meander:
                target_xs[1::2, :] = target_xs[1::2, ::-1]

            target_xys = np.stack([target_xs.flatten(), target_ys.flatten()], -1)
            target_zs = plane.predict(target_xys)

            pixel_indices = target_xys - np.array(origin) # make pixel indices relative to top-left corner
            pixel_indices *= [1, -1]

            area_targets = [
                (
                    tuple(px_index),
                    (x, y, z)
                ) for px_index, x, y, z in zip(pixel_indices, *target_xys.T, target_zs)
            ]
            
            area.targets = area_targets
            self.targets.extend(area_targets)


class EasyQueueAquisition(QueueAquisition):
    """
    Acquistion type that allows queueing of rectangular areas for successive acquisition.
    Works using a single plane generated from calibration points instead of using acquisition area-wise planes.
    """
    @_record_args
    def __init__(self, *args, **kwargs):
        self.queue = []
        self.plane = LinearRegression()
        super().__init__(*args, **kwargs)

    def calibrate(self, points):
        """
        Fit 3D plane to calibration points to generate z positions.
        :param points: calibration points, list of (x,y,z) tuples. Needs to have at least 3 points.
        """

        if len(points) < 3:
            raise ValueError("Calibration requires at least three points")

        points = np.asarray(points)
        
        self.plane.fit(points[:, :2], points[:, 2])
        r2 = self.plane.score(points[:, :2], points[:, 2])

        print(f"Fit r2 score: {r2:.3f}")

    def add_area(self, xy, width, height, step_size_x, step_size_y, name=None):
        """
        Create and add an AcquisitionArea to the queue.
        
        :param xy: - xy coordinates of the top-left corner. (x, y) tuple.
        :param width: - width of area.
        :param height: - height of area.
        """

        x, y = xy

        line_start = xy
        line_end = (x+width, y)
        perpendicular = (x, y-height) # NOTE super important to subtract here as y axis origin is on the bottom

        area = AcquistionArea(line_start, line_end, perpendicular, step_size_x, step_size_y, name=name)
        self.queue.append(area)
        
        return area
    
    @_record_args
    def generate_targets(self, meander=False):
        """
        Create the targets from queued areas.

        :param meander: scan each every-numbered row in reverse instead of jumping back to the row start (default False)
        
        """

        check_is_fitted(
            self.plane, 
            attributes="coef_", 
            msg="The acquisition series needs to be calibrated using the 'calibrate' function before generating targets!"
        )

        origin = self.image_bounds[1] # Pixel indices will be relative to top-left
        self.targets = []
        
        for area in self.queue:

            
            # Create XY coordinates for targets
            target_xs, target_ys = np.meshgrid(
                np.arange(
                    *np.sort([area.ls[0], area.le[0]]),
                    area.ssx),
                np.arange(
                    *np.sort([area.ls[1], area.pp[1]])[::-1], # invert since we scan top to bottom
                    area.ssy * -1
                )
            )

            if meander:
                target_xs[1::2, :] = target_xs[1::2, ::-1]

            target_xys = np.stack([target_xs.flatten(), target_ys.flatten()], -1)
            target_zs = self.plane.predict(target_xys)

            pixel_indices = target_xys - np.array(origin) # make pixel indices relative to top-left corner
            pixel_indices *= [1, -1]

            area_targets = [
                (
                    tuple(px_index),
                    (x, y, z)
                ) for px_index, x, y, z in zip(pixel_indices, *target_xys.T, target_zs)
            ]
            
            area.targets = area_targets
            self.targets.extend(area_targets)

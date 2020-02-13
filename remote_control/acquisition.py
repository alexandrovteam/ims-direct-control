import numpy as np
import json
from scipy import interpolate
from scipy.spatial.distance import euclidean, cosine
from skimage import measure

from remote_control.control import save_coords
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
        "circle": lambda xv, yv, r, c: np.square(xv - c[0])/((r[0]/2)**2) + np.square(yv - c[1])/((r[0]/2)** 2) < 1,
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

def unit_vector(v):
    return v / np.linalg.norm(v)

pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:, :-1]


def coord_formatter(data_x, data_y, data_z):
    data_x, data_y, data_z = map(np.array, [data_x, data_y, data_z])

    def format_coord(x,y):
        i = np.argmin((data_x - x) ** 2 + (data_y - y) ** 2)
        return 'Cursor: x=%#.5g, y=%#.5g\nNearest: x=%#.5g, y=%#.5g, z=%#.5g' % (x, y, data_x[i], data_y[i], data_z[i])

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


class Aquisition():
    def __init__(self, config_fn, datadir, log_fname=""):
        self.parse_config(config_fn)
        self.log_fname = log_fname
        self.datadir = datadir
        self.targets = []

    def parse_config(self, config_fn):
        self.config = json.load(open(config_fn))

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


    def set_image_bounds(self, image_bounds):
        self.image_bounds = image_bounds

    def acquire(self, dataset_name, dummy=True, measure=True):
        print("Acquiring {} ({} pixels)".format(dataset_name, len(self.targets)))
        xys = np.asarray([t[0] for t in self.targets])
        pos = np.asarray([t[1] for t in self.targets])
        acquire(
            self.config,
            self.log_fname,
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

    def plot_targets(self):
        import matplotlib.pyplot as plt
        safety_box = self.image_bounds
        xys = np.asarray([t[0] for t in self.targets])
        pos = [t[1] for t in self.targets]

        plt.figure()
        plt.plot([xy[0] for xy in xys], [xy[1] for xy in xys])
        plt.scatter([xy[0] for xy in xys], [xy[1] for xy in xys], s=3)
        plt.axis('equal')
        plt.title("Output coordinates")
        plt.gca().invert_yaxis()
        plt.show()

        plt.figure()
        plt.scatter([xy[0] for xy in pos], [xy[1] for xy in pos], c=[xy[2] for xy in pos], s=1)
        plt.plot(
            [safety_box[0][0], safety_box[0][0], safety_box[1][0], safety_box[1][0]],
            [safety_box[1][1], safety_box[0][1], safety_box[0][1], safety_box[1][1]],
            "--r"
        )
        plt.axis('equal')
        plt.title("Physical shape")
        plt.gca().invert_yaxis()
        plt.gca().format_coord = coord_formatter([xy[0] for xy in pos], [xy[1] for xy in pos], [xy[2] for xy in pos])
        plt.colorbar()
        plt.show()


class RectangularAquisition(Aquisition):
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


class WellPlateGridAquisition(Aquisition):
    def __init__(self, plate_type, *args, **kwargs):
        if isinstance(plate_type, dict):
            self.plate_type = plate_type['name']
            self.plate=plate_type
        else:
            self.plate_type = plate_type
            self.plate = SLIDES[plate_type]
        self.tform = [] #transformation matrix
        super().__init__(*args, **kwargs)

        self.subpattern_coords = [(0, 0)]
        self.subpattern_pixels = [(0, 0)]
        self.subpattern_grid_size = (1,1)
        self.subpattern_grid_offset = (1,1)

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
            "top_right":    lambda wellixs: spacing * wellixs + np.asarray([2*self.origin[0], 0, 0]),
            "bottom_left":  lambda wellixs: spacing * wellixs + np.asarray([0, 2*self.origin[1], 0]),
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
            [maxs[0], mins[1], 0],
            [mins[0], maxs[1], 0],
            [maxs[0], maxs[1], 0]
        ]
        locations = ["top_left", "top_right", "bottom_left", "bottom_right"]
        t = [self.transform(self.well_coord(e,l).reshape(1,-1))[0] for e,l in zip(extremes, locations)]
        return np.asarray(t)


    def generate_targets(self, wells_to_acquire, pixelsize_x, pixelsize_y,
                                    offset_x, offset_y,
                                    mask_function_name=None, area_function_name=None):

        pixelsize_x *= self.subpattern_grid_size[0]
        pixelsize_y *= self.subpattern_grid_size[1]

        if mask_function_name is None:
            print(self.plate)
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

        def make_subpattern(_x, _y):
            sx, sy = self.subpattern_grid_size
            ox, oy = self.subpattern_grid_offset
            for (sc_x, sc_y), (sp_x, sp_y) in zip(self.subpattern_coords, self.subpattern_pixels):
                pixel_x = (_x - x0) / pixelsize_x * sx + ox + sp_x
                pixel_y = (_y - y0) / pixelsize_y * sy + oy + sp_y
                pos_x = _x + offset_x + sc_x
                pos_y = _y + offset_y + sc_y
                pos_z = _z(pos_x, pos_y)[0]
                yield (pixel_x, pixel_y), (pos_x, pos_y, pos_z)

        mask_labels = measure.label(mask, background=0)
        self.targets = []
        for ii in range(1, np.max(mask_labels) + 1):
            _xy = list([
                coords
                for _x, _y in zip(xv[mask_labels == ii].flatten(), yv[mask_labels == ii].flatten())
                for coords in make_subpattern(_x, _y)
            ])
            self.targets.extend(_xy)

        print("total pixels: ", len(self.targets))

    def acquire_wells(self,
                      wells_to_acquire,
                      dataset_name,
                      dummy=True,
                      pixelsize_x=50, pixelsize_y=50,
                      offset_x = 0, offset_y=0,
                      area_shape = None,
                      area_mask = None,
                      ):
        """ DEPRECATED - use generate_targets and acquire separately instead
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
        self.acquire(dataset_name=dataset_name, dummy=dummy)




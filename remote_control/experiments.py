import numpy as np
import json
import datetime
import remote_control.control as rc
class RectangularImage():
    def __init__(self, origin, pixel_spacing, image_size):
        """
        
        :param origin: motor coordinates of start position (x,y,z) 
        :param pixel_spacing: distance between pixels (x,y)
        :param image_size: pixels to acquire (x,y)
        """
        self.origin = np.asarray(origin)
        self.pixel_spacing = np.asarray(pixel_spacing)
        self.image_size = np.asarray(image_size)
        self.initialise()

    def initialise(self):
        X, Y = np.meshgrid(np.arange(0, self.image_size[0]), np.arange(0, self.image_size[1]))
        X = X.flatten()
        Y = Y.flatten()
        xi = np.lexsort([X, Y])
        self.xys = np.vstack([X, Y]).T
        self.pos = np.vstack([
            self.origin[0] - X[xi]*self.pixel_spacing[0],
            self.origin[1] + Y[xi]*self.pixel_spacing[1],
            self.origin[2] + np.ones(X.shape)
                ]).T


    def acquire(self, data_fn, config_fn,  dummy=True, image_bounds = None):
        print "Acquiring {}".format(data_fn)
        xys = list(self.xys)
        pos = list(self.pos)
        ### ----- Config ---- ###
        self.log_fname = "./logfile_{}-{}".format(data_fn, datetime.datetime.now())
        fout = file(self.log_fname, 'w+')
        config = json.load(open(config_fn))
        HOST = config['host']
        user = config['user']
        password = config['password']

        ### --- Automatic Stuff ---###

        child = rc.initialise_and_login(HOST, user, password, fout, fout_r=None, fout_s=None)
        child.sendline('Begin')
        child.expect("OK")
        try:
            for xyz in pos:
                rc.acquirePixel(child, xyz, image_bounds, dummy=dummy)
        except Exception as e:
            print e
            raise
        child.sendline("End")
        child.close()
        coords_fname = "/media/embl/shared/directcontrolpositions/{}-{}.positions.json".format(data_fn, datetime.datetime.now().strftime("%y.%m.%d-%hh%mm%ss"))
        rc.save_coords(coords_fname, xys, pos, [], [])
        print 'done'
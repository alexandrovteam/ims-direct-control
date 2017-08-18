import pexpect
import numpy as np

class Control():
    def __init__(self, HOST, user, password, logfile = None, logfile_read = None, logfile_send = None):
        self.interface = self.initialise_and_login(HOST, user, password, logfile, logfile_read, logfile_send)
        self.image_bounds = None

    def initialise_and_login(self, HOST, user, password, fout, fout_r, fout_s):
        child = pexpect.spawn('telnet {}'.format(HOST))
        if fout:
            child.logfile = fout
        if fout_r:
            child.logfile_read = fout_r
        if fout_s:
            child.logfile_send = fout_s
        child.sendline("")  # the login is strange and puts characters on the command line - this fails one login so we're ready for the next
        child.sendline("")
        self.login(child, user, password)
        try:
            child.expect('Connected.')
        except:
            if child.expect("Anmeldung.") == 0:
                print 'login failed'
                self.login(child, user, password)
        return child


    def ix_to_pos(self, x, y, px_size, im_origin):
        return im_origin[0] - x * px_size[0], im_origin[1] + y * px_size[1], im_origin[2]


    def gotostr(self, xyz):
        return "Goto {};{};{}".format(xyz[0], xyz[1], xyz[2]).replace(".", ",")


    def check_image_bounds(self, xyz):
        x, y = xyz[0], xyz[1]
        image_bounds = self.image_bounds
        if not all([image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]]):
            print x, y, [image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]]
            raise IOError('Pixel out of bounding box {}'.format(image_bounds))


    def acquirePixel(self, child, xyz, ignore_bounds=False, dummy=False):
        if self.image_bounds == None and ignore_bounds==False:
            raise Exception('Image bounds not set')

        if ignore_bounds==False:
            self.check_image_bounds(xyz)

        if dummy:
            print self.gotostr(xyz)
            return 0
        else:
            child.sendline(self.gotostr(xyz))
            child.expect("OK")
            child.sendline('Meas')
            return child.expect("OK")

    def login(child, user, password):
        child.expect('Benutzername:')
        child.sendline(user)
        child.expect('Passwort:')
        child.sendline(password)
        # TODO return child.expect(<confirmation string?>)

    def set_image_bound(self, image_bounds):
        self.image_bounds = image_bounds

    def save_coords(self,json_fname, xys, pos, im_origin, pixel_size):
        import json
        info = {
            "index x": [xy[0] for xy in xys],
            "index y": [xy[1] for xy in xys],
            "pos x": [xy[0] for xy in pos],
            "pos y": [xy[1] for xy in pos],
            "pos z": [xy[2] for xy in pos],
            "pixel size": pixel_size,
            "im_origin": im_origin
        }
        json.dump(info, open(json_fname, 'w+'))
        return 0


    def ping(self):
        # Ping the instrument with a single pixel aquisition
        # child = initialise_and_login(HOST, user, password, fout)
        self.interface.sendline('Begin')
        self.interface.sendline('Meas')
        self.interface.sendline('End')
        self.interface.close()


    def runner(config):
        return NotImplementedError()
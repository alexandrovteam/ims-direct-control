import pexpect
import numpy as np

def initialise_and_login(HOST, user, password, fout, fout_r, fout_s):
    child = pexpect.spawnu('telnet {}'.format(HOST))
    if fout:
        child.logfile = fout
    if fout_r:
        child.logfile_read = fout_r
    if fout_s:
        child.logfile_send = fout_s
    child.sendline("")  # the login is strange and puts characters on the command line - this fails one login so we're ready for the next
    child.sendline("")
    login(child, user, password)
    try:
        child.expect('Connected.')
    except:
        if child.expect("Anmeldung.") == 0:
            login(child, user, password)
    return child


def ix_to_pos(x, y, px_size, im_origin):
    return im_origin[0] - x * px_size[0], im_origin[1] + y * px_size[1], im_origin[2]


def gotostr(xyz,maxtravel=2000):
    # TODO limit maximum single travel step
    return "Goto {};{};{}".format(xyz[0], xyz[1], xyz[2]).replace(".", ",")


def acquirePixel(child, xyz, image_bounds=None, dummy=False, maxtravel = 2000):
    if image_bounds:
        x, y = xyz[0], xyz[1]
        if not all([image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]]):
            print(x, y, [image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]])
            raise IOError('Pixel {} out of bounding box {}'.format((x,y), image_bounds))
    if dummy:
        print(gotostr(xyz))
        return 0
    child.sendline(gotostr(xyz))
    child.expect("OK")
    child.sendline('Meas')
    return child.expect("OK")


def login(child, user, password):
    child.expect('Benutzername:')
    child.sendline(user)
    child.expect('Passwort:')
    child.sendline(password)
    # TODO return child.expect(<confirmation string?>)


def save_coords(json_fname, xys, pos, im_origin, pixel_size):
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


def ping(child):
    # Ping the instrument with a single pixel aquisition
    # child = initialise_and_login(HOST, user, password, fout)
    child.sendline('Begin')
    child.sendline('Meas')
    child.sendline('End')
    child.close()


def runner(config):
    return NotImplementedError()
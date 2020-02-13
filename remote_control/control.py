import pexpect

def initialise_and_login(HOST, user, password, fout, fout_r=None, fout_s=None):
    child = pexpect.spawn('telnet {}'.format(HOST), encoding='utf-8')
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


def gotostr(xyz):
    # TODO limit maximum single travel step
    return "Goto {};{};{}".format(xyz[0], xyz[1], xyz[2]).replace(".", ",")


def get_position(child, autofocus=True):
    if autofocus:
        child.sendline('aflight 20')
        child.expect("OK")
        child.sendline('lights 0')
        child.expect("OK")
        child.sendline('focus')
        child.expect("OK")
    child.sendline('getpos')
    coord_line = child.readline()
    coord_strs = coord_line.strip().replace(',','.').split(';')
    coords = map(float, coord_strs)

    if autofocus:
        child.sendline('aflight 0')
        child.expect("OK")

    return coords



def acquirePixel(child, xyz, image_bounds=None, dummy=False, measure=True):
    if image_bounds:
        x, y = xyz[0], xyz[1]
        if not all([image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]]):
            print(x, y, [image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]])
            raise IOError('Pixel {} out of bounding box {}'.format((x,y), image_bounds))
    if dummy:
        print(gotostr(xyz))
    else:
        child.sendline(gotostr(xyz))
        child.expect("OK")
        if measure:
            child.sendline('Meas')
            child.expect("OK")


def login(child, user, password):
    child.expect('Benutzername:')
    child.sendline(user)
    child.expect('Passwort:')
    child.sendline(password)
    # TODO return child.expect(<confirmation string?>)


def save_coords(json_fname, xys, pos, im_origin, pixel_size):
    import json
    info = {
        "index x": [xy.item(0) for xy in xys],
        "index y": [xy.item(1) for xy in xys],
        "pos x": [xy.item(0) for xy in pos],
        "pos y": [xy.item(1) for xy in pos],
        "pos z": [xy.item(2) for xy in pos],
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
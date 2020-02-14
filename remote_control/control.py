import pexpect

def initialise_and_login(config):
    logfile = open(config['logfile'], 'w+')
    cmd = config["telnet"] + ' ' + config['host']
    if 'spawn' in dir(pexpect):
        # POSIX
        child = pexpect.spawn(cmd, encoding='utf-8', logfile=logfile)
    else:
        # Windows
        from pexpect.popen_spawn import PopenSpawn
        child = PopenSpawn(cmd, encoding='utf-8', logfile=logfile)

    child.sendline("")  # the login is strange and puts characters on the command line - this fails one login so we're ready for the next
    child.sendline("")
    login(child, config['user'], config['password'])
    try:
        child.expect('Connected.')
    except:
        if child.expect("Anmeldung.") == 0:
            login(child, config['user'], config['password'])
    return child


def ix_to_pos(x, y, px_size, im_origin):
    return im_origin[0] - x * px_size[0], im_origin[1] + y * px_size[1], im_origin[2]


def gotostr(xyz):
    # TODO limit maximum single travel step
    return "Goto {};{};{}".format(xyz[0], xyz[1], xyz[2]).replace(".", ",")


def get_position(child, autofocus=True, reset_light_to=255):
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
    if reset_light_to is not None:
        child.sendline(f'lights {reset_light_to}')

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
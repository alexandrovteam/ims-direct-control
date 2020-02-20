import sys, time
from io import TextIOWrapper
from telnetlib import Telnet
from datetime import datetime
from typing import Optional

telnet: Optional[Telnet] = None
logfile: Optional[TextIOWrapper] = None


class ExpectException(Exception):
    pass


def expect(expected, timeout=30):
    if isinstance(expected, str):
        expected = [expected.encode()]
    idx, match, data = telnet.expect(expected, timeout)
    data = str(data, 'utf-8')
    logfile.write(datetime.now().isoformat() + ': ' + data + '\r\n')
    if match is None:
        raise ExpectException(f'Did not receive expected "${expected} after {timeout} seconds')
    return data


def readline(timeout=5):
    data = telnet.read_until(b'\r\n', timeout)
    data = str(data, 'utf-8')
    logfile.write(datetime.now().isoformat() + ': ' + data + '\r\n')
    return data
    
    
def sendline(data):
    telnet.write(data.encode() + b'\r\n')
    logfile.write(datetime.now().isoformat() + ': ' + data + '\r\n')

    
def flush_output_buffer(delay=0.01):
    time.sleep(delay)
    data = telnet.read_eager()
    data = str(data, 'utf-8')
    logfile.write(datetime.now().isoformat() + ': ' + data + '\r\n')

    
def close(quit=False):
    global telnet, logfile
    if quit:
        sendline('Quit')
    telnet.close()
    telnet = None
    logfile = None
    
    
def initialise_and_login(config):
    global telnet, logfile
    if telnet is None:
        if config.get('logfile') == 'stdout':
            logfile = sys.stdout
        elif config.get('logfile'):
            logfile = open(config['logfile'], 'w+')
        else:
            logfile = None
        telnet = Telnet(config['host'])
        
        try:
            expect('Benutzername:')
            sendline(config['user'])
            expect('Passwort:')
            sendline(config['password'])
            result = readline()

            try:
                expect('Benutzername:', timeout=0.005)
                raise Exception('Login failed')
            except ExpectException:
                pass # No more login prompt = success
        except:
            telnet.close()
            telnet = None
            raise


def gotostr(xyz):
    # TODO limit maximum single travel step
    return "Goto {};{};{}".format(xyz[0], xyz[1], xyz[2])


def set_light(value):
    sendline(f'Lights {value}')
    expect("OK")
    flush_output_buffer(0)
    

def get_position(autofocus=True, reset_light_to=255):
    if autofocus:
        sendline('AfLaser 20')
        expect("OK")
        set_light(0)
        sendline('Focus')
        expect("OK")
    
    flush_output_buffer()
    sendline('GetPos')
    coord_line = readline()
    coord_strs = coord_line.strip().replace(';OK','').split(';')
    coords = tuple(map(float, coord_strs))

    if autofocus:
        sendline('AfLaser 0')
        expect("OK")
    if reset_light_to is not None:
        set_light(0)

    flush_output_buffer()
    return coords


def acquirePixel(xyz, image_bounds=None, dummy=False, measure=True):
    if image_bounds:
        x, y = xyz[0], xyz[1]
        if not all([image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]]):
            print(x, y, [image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]])
            raise IOError('Pixel {} out of bounding box {}'.format((x,y), image_bounds))
    if dummy:
        print(gotostr(xyz))
    else:
        sendline(gotostr(xyz))
        expect("OK\r\n")
        flush_output_buffer(0)
        if measure:
            sendline('Meas')
            expect("OK\r\n")
            flush_output_buffer(0)


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


    
def telnet_disconnect(child_):
    child.sendline('Quit')
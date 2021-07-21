import sys, time
from io import TextIOWrapper
from telnetlib import Telnet
from datetime import datetime
from typing import Optional
import numpy as np

telnet: Optional[Telnet] = None
logfile: Optional[TextIOWrapper] = None

long_move_distance = None
long_move_z = None
last_position = None

# SMALDIControl Remote Control Help:
# =========================================================
#  ? or Help      - show this help info
#  AfLaser i      - autofocus laser intensity in percent
#  Begin          - start measurement mode
#  End            - end measurement mode
#  Focus          - execute autofocus (Returns OK, FAIL or ERR)
#  Goto x;y;z     - go to position (Goto ;;1 changes z only)
#  Lights i       - intensity of LEDs in percent
#  Meas           - start a measurement at actual position
#  GetPos         - read actual position (x;y;z)
#  Quit           - quit remote control!


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
    global telnet, logfile, last_position
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

            last_position = get_position(False, None)

            try:
                expect('Benutzername:', timeout=0.005)
                raise Exception('Login failed')
            except ExpectException:
                pass # No more login prompt = success
        except:
            telnet.close()
            telnet = None
            raise


def configure_fly_at_fixed_z(distance=None, z=None):
    """
    Sets up logic so that if the stage needs to move further than `distance`, it does so at the specified `z` distance.
    Pass `None` values to disable
    """
    global long_move_distance, long_move_z
    long_move_distance = distance
    long_move_z = z


def goto(xyz, dummy):
    global last_position
    if dummy:
        print("Goto {};{};{}".format(xyz[0], xyz[1], xyz[2]))
    else:
        sendline("Goto {};{};{}".format(xyz[0], xyz[1], xyz[2]))
        expect("OK\r\n")
        flush_output_buffer(0)

    last_position = tuple(xyz)


def set_light(value):
    sendline(f'Lights {value}')
    expect("OK")
    flush_output_buffer(0)
    

def get_position(autofocus=True, reset_light_to=100):
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


def acquirePixel(xyz, dummy=False, measure=True):
    global last_position
    # Initialize last_position if it wasn't captured during login
    if last_position is None:
        last_position = tuple(xyz)
    # If the long move parameters are set, move away from the slide when moving more than the specified distance
    if long_move_distance is not None and long_move_z is not None:
        distance = np.linalg.norm(np.subtract(xyz[:2], last_position[:2]))
        if distance >= long_move_distance:
            goto((*last_position[:2], long_move_z), dummy)
            goto((*xyz[:2], long_move_z), dummy)

    goto(xyz, dummy)

    if not dummy and measure:
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

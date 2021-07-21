import numpy as np
import remote_control.control as rc
import json


def tranform_stage_optical(x, y, ratio, offset=[0, 0]):
    x = (np.asarray(x) * ratio[0]) - offset[0]
    y = (np.asarray(y) * ratio[1]) - offset[1]
    return x, y

def transform_stage_ion(x, y, origin, pixel_size):
    return (x-origin[0])/pixel_size[0], (y-origin[1])/pixel_size[1]


def decode_b64png(s, imshape):
    '''
    Converts a base64 encoded png into a numpy array
    :param s: base64 string
    :param imshape: tuple of image size: (x,y)
    :return: numpy array
    '''
    import base64
    r = base64.decodebytes(s)
    q = np.frombuffer(r.split('\n')[2], dtype=np.float32)
    return q.reshape(imshape)


def acquire(config, xys, pos, image_bounds, dummy, coords_fname, measure=True):
    if image_bounds:
        for x, y, *z in pos:
            if not all([image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]]):
                print(x, y, [image_bounds[0][0] > x > image_bounds[1][0], image_bounds[0][1] < y < image_bounds[1][1]])
                raise IOError('Pixel {} out of bounding box {}'.format((x,y), image_bounds))

    if not dummy:
        rc.save_coords(coords_fname, xys, pos, [], [])
        rc.initialise_and_login(config)
        try: 
            rc.sendline('Begin')
            rc.expect("OK")
            try:
                for i, xyz in enumerate(pos):
                    # Turn off light after 100 scans
                    if i > 100:
                        rc.set_light(0)
                    rc.acquirePixel(xyz, measure=measure)
            except Exception as e:
                print(e)
                raise

            rc.flush_output_buffer(3) # Wait before ending - it seems like a few commands get queued, and sending End immediately stops them
            rc.sendline("End")
        finally:
            # Turn off light after any acquisition
            rc.set_light(0)
            rc.close()
    else:
        try:
            for xyz in pos:
                rc.acquirePixel(xyz, dummy=True)
        except Exception as e:
            print(e)
            raise
    print('done')

    
def getpos(config_fn, log_fname=None, autofocus=True, reset_light_to=255):
    config = json.load(open(config_fn))
    rc.initialise_and_login(config)
    return rc.get_position(autofocus)


def stop_telnet(config_fn):
    config = json.load(open(config_fn))
    rc.initialise_and_login(config)
    return rc.close(quit=True)

def set_light(config_fn, value):
    config = json.load(open(config_fn))
    rc.initialise_and_login(config)
    rc.set_light(value)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
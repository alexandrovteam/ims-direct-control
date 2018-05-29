import numpy as np
import remote_control.control as rc


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


def acquire(config, log_fname, xys, pos, image_bounds, dummy, coords_fname):

    fout = open(log_fname, 'w+')
    if not dummy:
        child = rc.initialise_and_login(config['host'], config['user'], config['password'], fout,
                                    fout_r=None, fout_s=None)
        child.sendline('Begin')
        child.expect("OK")
        try:
            for xyz in pos:
                rc.acquirePixel(child, xyz, image_bounds, dummy=dummy)
        except Exception as e:
            print(e)
            raise
        child.sendline("End")
        child.close()
        rc.save_coords(coords_fname, xys, pos, [], [])
    else:
        try:
            for xyz in pos:
                child = ""
                rc.acquirePixel(child, xyz, image_bounds, dummy=dummy)
        except Exception as e:
            print(e)
            raise
    print('done')
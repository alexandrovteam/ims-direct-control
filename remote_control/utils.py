import numpy as np

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
    r = base64.decodestring(s)
    q = np.frombuffer(r.split('\n')[2], dtype=np.float32)
    return q.reshape(imshape)



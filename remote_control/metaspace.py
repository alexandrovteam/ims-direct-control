from sm_annotation_utils import sm_annotation_utils
import json
import pandas as pd
import os
import numpy as np
import datetime
import json

from remote_control.utils import acquire

class Experiment():
    def __init__(self, ds_id, rc_fn, fdr=0.1, database='HMDB', config=None, datadir = "."):
        self.connect_metaspace(config)
        self.init_dataset(ds_id)
        self.get_annotations(fdr, database)
        self.datadir = datadir
        self.config = json.load(open(rc_fn))


    def connect_metaspace(self, config):
        if config:
            self.sm = sm_annotation_utils.SMInstance(config)
        else:
            self.sm = sm_annotation_utils.SMInstance()

    def init_dataset(self, ds_id):
        self.ds = self.sm.dataset(id=ds_id)
        self.metadata = json.loads(self.ds.metadata.json)
        self.dataset_name = self.metadata['metaspace_options']['Dataset_Name']
        self.charge = 1 if self.metadata['MS_Analysis']["Polarity"] == 'Positive' else -1

    def get_annotations(self, fdr, database):
        ans = self.ds.annotations(fdr=fdr, database=database)
        self.annotations = ans
        self.images = []
        self.precursors = []
        for an in ans:
            # get image
            # get precursor
            ims = self.ds.isotope_images(an[0], an[1])
            self.images.append(ims[0])
            self.precursors.append(ims.peak(0))

    def generate_targets(self, pertarget= 20):
        import numpy as np
        import scipy.ndimage as ndimage
        print("{} targets will be generated".format(pertarget * len(self.images)))
        targets = {}
        mask = np.zeros(self.images[0].shape, dtype=float)
        sigma = 3
        for ii in range(pertarget):
            for im, mz, an in zip(self.images, self.precursors, self.annotations):
                fmask = ndimage.gaussian_filter(np.asarray(mask == mz, dtype=float), sigma=sigma, order=0)
                if fmask.max() > 0.:
                    fmask = fmask / (8. * fmask.max())
                fmask = 1. - fmask
                _mask = im * np.asarray(mask == 0., dtype=float) * fmask
                mix = np.argmax(_mask)
                y = mix / im.shape[1]
                x = mix % im.shape[1]
                mask[y, x] = mz
                targets[(x,y)] = [[x, y], [], an, mz]
        self.targets = [targets[t] for t in targets]
        print("{} targets generated".format(len(self.targets)))


    def pixel_to_motor(self, primary):
        # primary = [0,0], [1,1], [0,1], [1,0] corners of image
        primary = np.asarray(primary)
        y, x= self.images[0].shape
        self.get_transform([[0,0,0], [x,y,0], [0,y,0], [x,0,0]], primary)
        for target in self.targets:
            target[1] = self.transform(target[0] + [0,])

    def get_transform(self, primary, secondary):
        primary, secondary = map(np.asarray, [primary, secondary])
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        X = pad(primary)
        Y = pad(secondary)
        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        self.A, res, rank, s = np.linalg.lstsq(X, Y)

    def optimise_run(self):
        x = np.asarray([t[1][0] for t in self.targets])
        y = np.asarray([t[1][1] for t in self.targets])
        ix = np.lexsort([x, y])
        self.targets = [self.targets[ii] for ii in ix]


    def transform(self, vect):
        vect = np.asarray(vect)
        if vect.ndim>1:
            pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
            unpad = lambda x: x[:, :-1]
        else:
            pad = lambda x: np.concatenate([x, np.asarray([ 1.])])
            unpad = lambda x: x[0:-1]
        return unpad(np.dot(pad(vect), self.A))

    def write_inclusion_list(self, data_name = None, replaceadduct=None, replacepolarity=None):
        if not data_name:
            data_name = self.dataset_name
        header = ["Formula [M]", "Formula type", "Species", "CS [z]", "Polarity", "Start [min]", "End [min]", "(N)CE",
                  "(N)CE type", "MSX ID", "Comment"]
        # dummy =   "C9H7NO", "Chemical formula", " -H", "1", "Negative", "", "", "", "", "", ""

        inclusion = []
        ms2_mzs = []
        for target in self.targets:
            mf = target[2][0]
            adduct = target[2][1]
            polarity = self.metadata['MS_Analysis']['Polarity']
            if replaceadduct:
                adduct = replaceadduct
            if replacepolarity:
                polarity = replacepolarity
            inclusion.append(
                [mf, 'Chemical formula', adduct, "1", polarity, "", "", "", "", "", ""])
            ms2_mzs.append(target[3])
        df = pd.DataFrame.from_records(inclusion, columns=header)
        pth = os.path.join(self.datadir, "inclusion_list_{}.csv".format(data_name).replace("/", "_"))
        df.to_csv(pth, sep=',', index=False)
        s = pd.Series(ms2_mzs)
        spth = os.path.join(self.datadir, "inclusion_list_mzs_{}.csv".format(data_name).replace("/", "_"))
        s.to_csv(spth)
        return pth, spth

    @property
    def coords_fname(self):
        return os.path.join(self.datadir, "{}-{}.positions.json".format(self.dataset_name, datetime.datetime.now().strftime("%Y%m%d-%hh%mm%ss")))

    @property
    def log_fname(self):
        return os.path.join(self.datadir, "logfile_{}-{}".format(self.dataset_name, datetime.datetime.now().strftime("%Y%m%d-%hh%mm%ss")))

    def acquire(self,  dummy=True, image_bounds = None):
        print("Acquiring {}".format(self.dataset_name))
        xys = np.asarray([t[0] for t in self.targets])
        pos = np.asarray([t[1] for t in self.targets])
        acquire(self.config, self.log_fname, xys, pos, image_bounds, dummy, self.coords_fname)

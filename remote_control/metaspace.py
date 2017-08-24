from sm_annotation_utils import sm_annotation_utils
import json
import pandas as pd
import os
import numpy as np
import remote_control.control as rc

class Experiment():
    def __init__(self, ds_id, fdr=0.1, database='HMDB', config=None):
        self.connect_metaspace(config)
        self.init_dataset(ds_id)
        self.get_annotations(fdr, database)


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
        print("{} targets will be generated".format(pertarget*len(self.images)))
        self.targets = []
        mask = np.ones(self.images[0].shape, dtype=float)
        for ii in range(pertarget):
            for ix in np.random.permutation(range(0, len(self.images))):
                im, mz, an = self.images[ix], self.precursors[ix], self.annotations[ix]
                mix = np.argmax(im*mask)
                y = mix / im.shape[1]
                x = mix % im.shape[1]
                #print(np.max(im*mask), (im*mask)[y,x], mask[y,x], mz)
                mask[y,x] = 0.
                #print(mask.flatten()[mix])
                self.targets.append([[x, y], [], an, mz])

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

    def write_inclusion_list(self, dr, data_name = None):
        if not data_name:
            data_name = self.dataset_name
        header = ["Formula [M]", "Formula type", "Species", "CS [z]", "Polarity", "Start [min]", "End [min]", "(N)CE",
                  "(N)CE type", "MSX ID", "Comment"]
        # dummy =   "C9H7NO", "Chemical formula", " -H", "1", "Negative", "", "", "", "", "", ""

        inclusion = []
        ms2_mzs = []
        for target in self.targets:
            inclusion.append(
                [target[1][0], 'Chemical formula', target[1][1], "1", self.metadata['MS_Analysis']['Polarity'], "", "", "", "", "", ""])
            ms2_mzs.append(target[2])
        df = pd.DataFrame.from_records(inclusion, columns=header)
        pth = os.path.join(dr, "inclusion_list_{}.csv".format(data_name).replace("/", "_"))
        df.to_csv(pth, sep=',', index=False)
        s = pd.Series(ms2_mzs)
        spth = os.path.join(dr, "inclusion_list_mzs_{}.csv".format(data_name).replace("/", "_"))
        s.to_csv(spth)
        return pth, spth


    def acquire(self, rc_fn,  dummy=True, image_bounds = None):
        import json
        import datetime
        print "Acquiring {}".format(self.dataset_name)
        xys = np.asarray([t[0] for t in self.targets])
        pos = np.asarray([t[1] for t in self.targets])
        ### ----- Config ---- ###
        self.log_fname = "./logfile_{}-{}".format(self.dataset_name, datetime.datetime.now())
        fout = file(self.log_fname, 'w+')
        config = json.load(open(rc_fn))
        HOST = config['host']
        user = config['user']
        password = config['password']

        ### --- Automatic Stuff ---###

        child = rc.initialise_and_login(HOST, user, password, fout, fout_r=None, fout_s=None)
        child.sendline('Begin')
        child.expect("OK")
        try:
            for xyz in pos:
                rc.acquirePixel(child, xyz, image_bounds, dummy=dummy)
        except Exception as e:
            print e
            raise
        child.sendline("End")
        child.close()
        coords_fname = "/media/embl/shared/directcontrolpositions/{}-{}.positions.json".format(self.dataset_name, datetime.datetime.now())
        rc.save_coords(coords_fname, xys, pos, [], [])
        print 'done
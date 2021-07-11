
import h5py
import glob
import numpy as np
import dxtbx


class Loader:

    def __init__(self):
        self.idx = None
        self.num_images = None
        self.DET = None  # dxtbx detector

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, val):
        if val is not None:
            if not isinstance(val, int):
                raise TypeError("index must be int!")

            if val < 0:
                raise ValueError("index must be positive!")

            if self.num_images is not None and val >= self.num_images:
                raise ValueError("index must be lower than number of images (%d)" % self.num_images)

        self._index = val

    @property
    def num_images(self):
        return self._num_images

    @num_images.setter
    def num_images(self, val):
        if val is not None:
            if not isinstance(val, int):
                raise TypeError("num images must be an integer!")
        self._num_images = val

    def __getitem__(self, index):
        self.index = index
        return self._image_from_index()

    def _image_from_index(self):
        pass


class DxtbxLoader(Loader):

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.dxtbx_loader = dxtbx.load(filename)
        self.num_images = self.dxtbx_loader.get_num_images()
        self.index = 0
        self.DET = self.dxtbx_loader.get_detector(self.index)

    def _image_from_index(self):
        image = self.dxtbx_loader.get_raw_data(self.index)
        image = np.array([i.as_numpy_array() for i in image])
        return image


class StageTwoBeforeAfterLoader(Loader):

    def __init__(self, filenames_glob, detector, image_shape=(256,254,254)):
        super().__init__()
        self.filenames = glob.glob(filenames_glob)
        self.num_images = len(self.filenames)
        print("Found %d filenames from glob" % self.num_images)
        self.index = 0
        # TODO add detector dictionary to stage two hdf5 file
        self._set_detector(detector)
        self.pids = None  # panel id of pixel
        self.xs = None  # fast scan coord of pix
        self.ys = None  # slow scan coord of pixel
        self.Z = None  # Z score for pixel

    def _set_detector(self, detector):
        self.DET = detector
        num_pan = len(self.DET)
        fast_dim, slow_dim = self.DET[0].get_image_size()
        self.image_shape = num_pan, slow_dim, fast_dim

    def _image_from_index(self):
        self.filename = self.filenames[0]
        self._load_h5()
        image = np.zeros(self.image_shape)
        image[self.pids, self.ys, self.xs] = self.Z
        return image

    def _load_h5(self):
        with h5py.File(self.filename, "r") as h5:
            keys = list(h5.keys())
            for k in "pids", "xs", "ys", "Z_model_noise":
                if k not in keys:
                    raise KeyError("File %s needs %s key" % (self.filename, k))
            self._load_data(h5)

    def _load_data(self, h5):
        self.pids = h5["pids"][()]
        self.xs = h5["xs"][()]
        self.ys = h5["ys"][()]
        self.Z = h5["Z_model_noise"][()]
        assert self.pids.shape == self.xs.shape == self.ys.shape == self.Z.shape


if __name__ == "__main__":
    dxtbx_loader = DxtbxLoader("/Users/dermen/gain_pix2.h5")
    image = dxtbx_loader[0]
    print("Image shape is", image.shape)

    # test indexing errors
    for i in [-1, dxtbx_loader.num_images]:
        try: dxtbx_loader[i]
        except ValueError as err:
            print("Caught error: message= %s" % err )
            pass
    try: dxtbx_loader["help"]
    except TypeError as err:
        print("Caught error: message= %s" % err )
        pass

    stage2_loader = StageTwoBeforeAfterLoader("/Users/dermen/aconcagua_1_ens/*after.h5", detector=dxtbx_loader.DET)

    print("OK!")




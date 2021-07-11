from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="path to image or a glob of images")
parser.add_argument("--detector", type=str, default=None, help="path to detector file")
parser.add_argument("--input_type", type=str, default="dxtbx", choices=["dxtbx", "stage_two_before_after"],
                    help="type of loader")
args = parser.parse_args()

import matplotlib as mpl
from dxtbx.model.experiment_list import ExperimentListFactory
import pylab as plt
import numpy as np
from range_slider import RangeSlider
import loader
from helper_widgets import LabeledEntry


def add_asic_to_ax(ax, I, p, fs, ss=None, s="", **kwargs):
    """
    View along the Z-axis (usually the beam axis) at the detector

    vectors are all assumed x,y,z
    where +x is to the right when looking at detector
          +y is to down when looking at detector
          z is along cross(x,y) 
   
    Note: this assumes slow-scan is prependicular to fast-scan

    Args
    ====
    ax, matplotlib axis
    I, 2D np.array
        panels panel
    p, corner position of first pixel in memory
    fs, fast-scan direction in lab frame
    ss, slow-scan direction in lab frame, 
    s , some text
    """
    # first get the angle between fast-scan vector and +x axis
    ang = np.arccos(np.dot(fs, [1, 0, 0]) / np.linalg.norm(fs))
    ang_deg = ang * 180 / np.pi
    if fs[0] <= 0 and fs[1] < 0:
        ang_deg = 360 - ang_deg
    elif fs[0] >= 0 and fs[1] < 0:
        ang_deg = 360 - ang_deg

    im = ax.imshow(I,
                   extent=(p[0], p[0] + I.shape[1], p[1] + I.shape[0], p[1]),
                   **kwargs)
    trans = mpl.transforms.Affine2D().rotate_deg_around(p[0], p[1], ang_deg) + ax.transData
    im.set_transform(trans)

    # add label to the axis
    panel_cent = .5 * fs * I.shape[1] + .5 * ss * I.shape[0] + p
    _text = ax.text(panel_cent[0], panel_cent[1], s=s, color='c')


import tkinter as tk


class VIEW:

    def __init__(self, input, input_type="dxtbx", detector=None):
        self.data = None
        self.loader = None

        self.setup_ax()
        self.load_inputs(input, input_type, detector)
        self.make_psf()
        self.init_plot()

    def setup_ax(self):
        plt.figure()
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-2500, 2500)  # work in pixel units
        self.ax.set_ylim(2500, -2500)
        self.ax.set_facecolor('dimgray')
        self.imshow_arg = {"interpolation": 'none', "cmap": 'gnuplot',
                           "vmin": 17.33, "vmax": 43.67}

        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)  # A tk.DrawingArea.
        # self.canvas.draw()
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        # self.toolbar.update()
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def load_inputs(self, input, input_type, detector):
        # fname = "/Users/dermen/gain_pix2.h5"
        if input_type == "dxtbx":
            self.loader = loader.DxtbxLoader(input)
            # self.loader.mask = pickle.load(open("/Users/dermen/resmask_2p1_to_2p5.pkl","rb"))
            # self.loader.mask = np.array([m.as_numpy_array() for m in self.mask])
            self.using_mask = False  # True
            # self.all_images = h5py.File(fname, "r")["images"][()]

        if input_type == "stage_two_before_after":
            detector = ExperimentListFactory.from_json_file(detector, False)[0].detector
            self.loader = loader.StageTwoBeforeAfterLoader(input, detector)
            self.using_mask = False

        self.data = self.loader[0]

    def make_psf(self):
        self.P, self.S, self.F = [], [], []
        for i in range(len(self.loader.DET)):
            panel = self.loader.DET[i]
            origin = np.array(panel.get_origin())
            fdet = np.array(panel.get_fast_axis())
            # fdet = np.array([fdet[0], fdet[1], 0])
            sdet = np.array(panel.get_slow_axis())
            # sdet = np.array([sdet[0], sdet[1], 0])
            # fdet /= np.linalg.norm(fdet)
            # sdet /= np.linalg.norm(sdet)
            pixsize = panel.get_pixel_size()[0]
            self.P.append(origin / pixsize)
            self.S.append(sdet)
            self.F.append(fdet)

    def init_plot(self):
        d = self.data.copy()

        if self.using_mask:
            d[~self.loader.mask] = np.nan
        for i in range(len(self.loader.DET)):
            add_asic_to_ax(ax=self.ax, I=d[i],
                           p=self.P[i], fs=self.F[i], ss=self.S[i], s="", **self.imshow_arg)
        plt.colorbar(self.ax.images[0])
        plt.draw()
        plt.pause(0.1)
        self.update_plot()  # for consistency

    def set_data(self, idx):
        if idx != self.loader.index:
            print("Moving to image %d / %d" % (idx + 1, self.loader.num_images))
            self.data = self.loader[idx]
            self.update_plot()
        else:
            print("Staying at image %d / %d" % (self.loader.index + 1, self.loader.num_images))

    def update_plot(self):
        d = self.data.copy()
        if self.using_mask:
            d[~self.loader.mask] = np.nan
        for i in range(len(self.loader.DET)):
            im = self.ax.images[i]
            im.set_data(d[i])
            self.ax.draw_artist(im)
        self.ax.set_title(
            "Image %d / %d (index=%d)" % (self.loader.index + 1, self.loader.num_images, self.loader.index))
        self.fig.canvas.blit(self.ax.bbox)


class CONTROL(tk.Frame):

    def __init__(self, root, VIEW, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.master = root
        self.VIEW = VIEW
        self.times_updated_clim = 0

        self._make_slider_frame()
        self._make_image_nav_frame()
        self._make_frame_for_going_to_image()
        self.vmin = self.vmax = None
        # self.key_bindings()
        self._update_clim(init=True)

    def _make_image_nav_frame(self):
        button_frame = tk.Frame(self.slider_frame)
        button_frame.pack()
        tk.Button(button_frame, text="Previous", command=self._on_button_prev, relief=tk.RAISED).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Next", command=self._on_button_next, relief=tk.RAISED).pack(side=tk.LEFT)

    def _make_slider_frame(self):
        self.slider_frame = tk.Toplevel()
        clim = self.VIEW.ax.images[0].get_clim()
        self.slider = RangeSlider(self.slider_frame, clim, color="#00fa32")
        self.slider.pack(fill=tk.BOTH, expand=tk.YES)

    def _make_frame_for_going_to_image(self):
        self.go_frame = tk.Frame(self.slider_frame)
        self.go_frame.pack()
        self._go_to_image_entry = LabeledEntry(self.go_frame,
                                               labeltext="Go to image (%d-%d)" % (0, self.VIEW.loader.num_images - 1),
                                               init_value=0, vartype=int)
        self._go_to_image_entry.pack(side=tk.LEFT)
        tk.Button(self.go_frame, text="Go!", command=self._go_to_index, relief=tk.RAISED).pack(side=tk.LEFT)
        self._go_to_image_entry.entry.bind("<Return>", self._go_to_index_on_return)

    def _go_to_index_on_return(self, event):
        self._go_to_index()

    def key_bindings(self):
        self.master.bind_all("<Shift-Left>", self._on_left)
        self.master.bind_all("<Shift-Right>", self._on_right)
        self.master.bind_all("<Shift-Down>", self._on_m)

    def _on_button_next(self):
        self._on_right(None)

    def _on_button_prev(self):
        self._on_left(None)

    def _on_m(self, event):
        self.VIEW.using_mask = not self.VIEW.using_mask
        if self.VIEW.using_mask:
            print("Using mask!")
        else:
            print("not Using mask!")

        self.VIEW.update_plot()

    def _go_to_index(self):
        new_idx = self._go_to_image_entry.variable.get()
        if new_idx >= self.VIEW.loader.num_images:
            print("WARNING: max image index is %d" % (self.VIEW.loader.num_images - 1))
            new_idx = self.VIEW.loader.num_images - 1
        if new_idx >= self.VIEW.loader.num_images:
            print("WARNING: min image index is 0")
            new_idx = 0
        self.VIEW.set_data(new_idx)

    def _on_right(self, event):
        new_idx = min(self.VIEW.loader.index + 1, self.VIEW.loader.num_images - 1)
        print(new_idx)
        self.VIEW.set_data(new_idx)

    def _on_left(self, event):
        new_idx = max(self.VIEW.loader.index - 1, 0)
        print(new_idx)
        self.VIEW.set_data(new_idx)

    def _update_clim(self, init=False):
        new_vmin, new_vmax = self.slider.minval, self.slider.maxval
        if init:
            self.vmin = new_vmin
            self.vmax = new_vmax

        clim_has_changed = new_vmin != self.vmin or new_vmax != self.vmax
        if init or clim_has_changed:
            print("Updating clim %d times" % self.times_updated_clim)
            for i in range(len(self.VIEW.loader.DET)):
                im = self.VIEW.ax.images[i]
                im.set_clim(vmin=new_vmin, vmax=new_vmax)
                self.VIEW.ax.draw_artist(im)
            self.VIEW.fig.canvas.blit(self.VIEW.ax.bbox)
            self.vmin = new_vmin
            self.vmax = new_vmax
            self.times_updated_clim += 1

        self.master.after(1000, self._update_clim)


root = tk.Tk()
V = VIEW(args.input, args.input_type, args.detector)
C = CONTROL(root, V)
C.pack(fill=tk.BOTH, expand=tk.YES)
tk.mainloop()

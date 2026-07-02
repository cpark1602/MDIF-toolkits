#!/usr/bin/env python
# coding: utf-8

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as pl
import scipy.constants
import scipy.stats 
import os

import time
import re

import warnings
warnings.filterwarnings(action='once')

import logging
from MDAnalysis import MissingDataWarning, NoDataError, SelectionError, SelectionWarning
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from MDAnalysis.lib import distances
logger = logging.getLogger('MDAnalysis.analysis.hbonds')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #print('idx: ', idx, array[idx])
    return idx


def read_file_x_y(filename):
        #open_file=argv[1]
        x = []
        y = []
        dy = []
        txt = open(filename)
        while True:
                line = txt.readline().strip('\n')
                #print line
                if len(line) == 0:
                        break
                else:
                        line = line.split()
                        x.append(float(line[0]))
                        y.append(float(line[1]))
                        #dy.append(float(line[2]))
        txt.close()
        return x, y     #, dy


class MSD:
    def __init__(self, universe, select='all', msd_type='xyz', fft=True,  start=None, stop=None, step=None, **kwargs): 
        
        self.u = universe
        # args
        self.select = select
        self.msd_type = msd_type
        self.fft = fft

        # local
        self.ag = u.select_atoms(self.select)
        self.n_particles = len(self.ag)
        print('n particle: ', self.n_particles)
        self._position_array = None

        # result
        self.msds_by_particle = None
        self.timeseries = None

        self._parse_msd_type()
        self._setup_frames(self.u.trajectory, start=start, stop=stop, step=step)

    def _parse_msd_type(self):
        """ Sets up the desired dimensionality of the MSD.
        """
        keys = {'x': [0], 'y': [1], 'z': [2], 'xy': [0, 1],
                'xz': [0, 2], 'yz': [1, 2], 'xyz': [0, 1, 2]}

        self.msd_type = self.msd_type.lower()

        try:
            self._dim = keys[self.msd_type]
        except KeyError:
            raise ValueError(
                'invalid msd_type: {} specified, please specify one of xyz, '
                'xy, xz, yz, x, y, z'.format(self.msd_type))

        self.dim_fac = len(self._dim)

    def _prepare(self):
        # self.n_frames only available here
        # these need to be zeroed prior to each run() call
        self.msds_by_particle = np.zeros((self.n_frames,
                                                  self.n_particles))
        self._position_array = np.zeros(
            (self.n_frames, self.n_particles, self.dim_fac))
        # self.timeseries not set here

    def _setup_frames(self, trajectory, start=None, stop=None, step=None):
        """
        Pass a Reader object and define the desired iteration pattern
        through the trajectory

        Parameters
        ----------
        trajectory : mda.Reader
            A trajectory Reader
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame


        .. versionchanged:: 1.0.0
            Added .frames and .times arrays as attributes

        """
        self._trajectory = trajectory
        start, stop, step = trajectory.check_slice_indices(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step
        self.n_frames = len(range(start, stop, step))
        self.frames = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

    def _parse_msd_type(self):
        r""" Sets up the desired dimensionality of the MSD.

        """
        keys = {'x': [0], 'y': [1], 'z': [2], 'xy': [0, 1],
                'xz': [0, 2], 'yz': [1, 2], 'xyz': [0, 1, 2]}

        self.msd_type = self.msd_type.lower()

        try:
            self._dim = keys[self.msd_type]
        except KeyError:
            raise ValueError(
                'invalid msd_type: {} specified, please specify one of xyz, '
                'xy, xz, yz, x, y, z'.format(self.msd_type))

        self.dim_fac = len(self._dim)

    def _single_frame(self):
        r""" Constructs array of positions for MSD calculation.

        """
        # shape of position array set here, use span in last dimension
        # from this point on
        self._position_array[self._frame_index] = (
            self.ag.positions[:, self._dim])

    def _conclude(self):
        if self.fft:
            self._conclude_fft()
        else:
            self._conclude_simple()

    def _conclude_simple(self):
        r""" Calculates the MSD via the simple "windowed" algorithm.

        """
        lagtimes = np.arange(1, self.n_frames)
        positions = self._position_array.astype(np.float64)
        for lag in lagtimes:
            disp = positions[:-lag, :, :] - positions[lag:, :, :]        # positions[frames, atoms, xyz]
            sqdist = np.square(disp).sum(axis=-1)                        # sum is for x^2 + y^2 + z^2
            self.msds_by_particle[lag, :] = np.mean(sqdist, axis=0)
        self.timeseries = self.msds_by_particle.mean(axis=1)

    def _conclude_fft(self):  # with FFT, np.float64 bit prescision required.
        r""" Calculates the MSD via the FCA fast correlation algorithm.

        """
        try:
            import tidynamics
        except ImportError:
            raise ImportError("""ERROR --- tidynamics was not found!

                tidynamics is required to compute an FFT based MSD (default)

                try installing it using pip eg:

                    pip install tidynamics

                or set fft=False""")

        positions = self._position_array.astype(np.float64)
        for n in range(self.n_particles):
            self.msds_by_particle[:, n] = tidynamics.msd(
                positions[:, n, :])
        self.timeseries = self.msds_by_particle.mean(axis=1)

    def run(self, **kwargs):

        self._prepare()
        i = 0
        for ts in self.u.trajectory[self.start:self.stop:self.step]:
            self._frame_index = i
            self._ts = ts
            self.frames[i] = ts.frame
            self._single_frame()
            i += 1
        self._conclude()
        return self


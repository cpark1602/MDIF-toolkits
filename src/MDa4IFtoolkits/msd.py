#!/usr/bin/env python
# coding: utf-8

import warnings
import logging
import numpy as np

# Setup warning filters and logger configuration to avoid excessive stdout noise
warnings.filterwarnings(action="once")
logger = logging.getLogger("MDAnalysis.analysis.hbonds")


def find_nearest(array, value):
    """
    Finds the index of the element in an array closest to a specified value.
    Uses an absolute difference minimization strategy.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def read_file_x_y(filename):
    """
    Parses a plain-text tabular file extracting the first two numeric 
    whitespace-separated columns into decoupled Python lists (e.g., X and Y).
    """
    x = []
    y = []
    txt = open(filename)
    while True:
        line = txt.readline().strip("\n")
        if len(line) == 0:  # Break out at EOF (End of File)
            break
        else:
            line = line.split()
            x.append(float(line[0]))
            y.append(float(line[1]))
    txt.close()
    return x, y


class MSD:
    """
    Computes Mean Squared Displacement (MSD) for an MDAnalysis atom group.
    Supports basic windowed looping or Fast Fourier Transform (FFT) methods via tidynamics.
    """
    def __init__(
        self,
        universe,
        select="all",
        msd_type="xyz",
        fft=True,
        start=None,
        stop=None,
        step=None,
    ):
        self.u = universe
        self.select = select
        self.msd_type = msd_type
        self.fft = fft

        # --- Dynamic Selection Mapping ---
        self.ag = self.u.select_atoms(self.select)
        self.n_particles = len(self.ag)
        print("n particle: ", self.n_particles)
        
        # Internal cache tracking raw structural configurations over time
        self._position_array = None

        # --- Storage Result Vectors ---
        self.msds_by_particle = None  # 2D tracking matrix: [frames, particles]
        self.timeseries = None        # 1D array profile representing system average MSD

        # --- Coordinate Parsing & Frame Window Range Slicing ---
        self._parse_msd_type()
        self._setup_frames(self.u.trajectory, start=start, stop=stop, step=step)

    def _parse_msd_type(self):
        """
        Parses the directional code to map Cartesian target index keys 
        and scale projection operations.
        """
        keys = {
            "x": [0],
            "y": [1],
            "z": [2],
            "xy": [0, 1],
            "xz": [0, 2],
            "yz": [1, 2],
            "xyz": [0, 1, 2],
        }

        self.msd_type = self.msd_type.lower()

        try:
            self._dim = keys[self.msd_type]
        except KeyError:
            raise ValueError(
                "invalid msd_type: {} specified, please specify one of xyz, "
                "xy, xz, yz, x, y, z".format(self.msd_type)
            )

        self.dim_fac = len(self._dim)

    def _prepare(self):
        """
        Initializes multi-dimensional zero arrays once the absolute frame count 
        bounds are determined.
        """
        self.msds_by_particle = np.zeros((self.n_frames, self.n_particles))
        self._position_array = np.zeros((self.n_frames, self.n_particles, self.dim_fac))

    def _setup_frames(self, trajectory, start=None, stop=None, step=None):
        """
        Configures user frame selection patterns with MDAnalysis native index wrapping checks.
        """
        self._trajectory = trajectory
        start, stop, step = trajectory.check_slice_indices(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step
        
        # Calculate true trajectory tracking spans
        self.n_frames = len(range(start, stop, step))
        self.frames = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

    def _single_frame(self):
        """
        Populates the primary history array with spatial coordinates from the current frame index.
        """
        self._position_array[self._frame_index] = self.ag.positions[:, self._dim]

    def _conclude(self):
        """
        Route calculation data pipeline out to processing handlers.
        """
        if self.fft:
            self._conclude_fft()
        else:
            self._conclude_simple()

    def _conclude_simple(self):
        """
        Calculates the MSD using a classic sliding-window approach over lag times.
        """
        lagtimes = np.arange(1, self.n_frames)
        positions = self._position_array.astype(np.float64)  # Ensure precision
        
        for lag in lagtimes:
            # Shift coordinate arrays by the current lag step
            disp = (
                positions[:-lag, :, :] - positions[lag:, :, :]
            )  # Array structure layout: [frames, atoms, coordinates]
            
            # Compute squared displacement across active projection axes
            sqdist = np.square(disp).sum(axis=-1)  
            
            # Record average displacements calculated across time windows
            self.msds_by_particle[lag, :] = np.mean(sqdist, axis=0)
            
        # Compress tracking dimensions out into an ensemble system average
        self.timeseries = self.msds_by_particle.mean(axis=1)

    def _conclude_fft(self):
        """
        Calculates individual atom profiles via the Fast Correlation Algorithm (FCA) using FFTs.
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
        
        # Process individual atom paths using external high-performance routines
        for n in range(self.n_particles):
            self.msds_by_particle[:, n] = tidynamics.msd(positions[:, n, :])
            
        # Evaluate collective average across particles
        self.timeseries = self.msds_by_particle.mean(axis=1)

    def run(self):
        """
        Main execution manager looping through target trajectory boundaries 
        to cache positions before triggering final calculations.
        """
        self._prepare()
        i = 0
        for ts in self.u.trajectory[self.start : self.stop : self.step]:
            self._frame_index = i
            self._ts = ts
            self.frames[i] = ts.frame
            self._single_frame()
            i += 1
            
        self._conclude()
        return self

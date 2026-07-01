#!/usr/bin/env python

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as pl
import scipy.constants
import scipy.stats 
import os
import shutil    # copy file
import time
import re

import warnings
warnings.filterwarnings(action='once')

#import matplotlib.pyplot as pl
#from matplotlib import rc
#import matplotlib.ticker as ticker
#from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
#import matplotlib.font_manager as fm
#font_names = [f.name for f in fm.fontManager.ttflist]
# # Hydrogen bond analysis
import warnings
import logging
from MDAnalysis import MissingDataWarning, NoDataError, SelectionError, SelectionWarning
from MDAnalysis.lib.log import ProgressBar
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from MDAnalysis.lib import distances
from MDAnalysis.lib.correlations import autocorrelation, correct_intermittency
logger = logging.getLogger('MDAnalysis.analysis.hbonds')

#####----- Trj path -----
w_path="./"
#from sys import argv
# Use only xyz format.
#trj_file1=argv[1]
u_if = mda.Universe('run-pos.pdb', 'run-pos.dcd')


print("total nr. of frame: ", len(u_if.trajectory))
tot_frames = len(u_if.trajectory)
boxX = 48.57; boxY = 15.667; boxZ = 15.076
#boxX = 23.0000; boxY = 23.0000; boxZ = 22.3404
box = [boxX, boxY, boxZ, 90, 90, 90]
u_if.dimensions = box
"Velesco angle H-O-O 35 degrees"
HBs_criteria_input = 'Sho'   # Luzar: a rectangule; Sho: Triangle
#start_stop_step = [0, tot_frames, 1]  # q0.0-region2-new/ddec
#start_stop_step = [0, 10000, 1]  # q0.0-region2-new/ddec
start_stop_step = [0, 3, 1]  # q0.0-region2-new/ddec

chemisorbed_cutoff_O = 10.5
chemisorbed_cutoff_H = 9.85
path_results='./results/atomic_charge/'   # where the NAC results are stored
print_results_path=w_path+'/results-acf-cdot-xyz/'     # To save the results

import acf
if_q0_nac = acf.ACF(u_if, box, HBs_criteria_input, 'name O', 'name O', print_results_path, cutoff_dist_O_H =3.5, cutoff_dist_donor_acceptor = 3.5, cutoff_IF = [0, 12], cutoff_BULK = [19, 28], angle=35.0, pbc=True, start=start_stop_step[0], stop=start_stop_step[1], step=start_stop_step[2], nac='IF')  #tot_frames
if_q0_nac.run()

print('Use xyz format')
print('Ne cutoff for the HB ACF analysis is 31 AA')


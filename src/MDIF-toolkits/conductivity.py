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

#----------------------------------
# Path
#----------------------------------
r_path="./results"

#import warnings
#warnings.filterwarnings(action='once')

#import matplotlib.pyplot as pl
#from matplotlib import rc
#import matplotlib.ticker as ticker
#from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
#import matplotlib.font_manager as fm
#font_names = [f.name for f in fm.fontManager.ttflist]

class Conductivity:
    def __init__(self, universe, selection1, selection2, print_results_path, Ex, start=None, stop=None, step=None, **kwargs):   # 
        
        self.u = universe
        self.seconds1 = time.time()
        self.t_l = []

        self.total_frame = len(self.u.trajectory)
        self.selection1 = selection1
        self.selection2 = selection2
        self.n_frames = None
        self.start = start
        self.stop = stop
        self.step = step
        self.Ex = Ex

    def _compute(self, ion1_av_vel_x, density_ion1, z_ion1):
        aver_ion1 = np.mean(ion1_av_vel_x)  #sum_vel_ion1/n_ion1
        aver_err_ion1 = np.std(ion1_av_vel_x)  # aver_err(vel_x_ion1,aver_ion1)
        
        vel_ion1=aver_ion1	# [nm/ns] 
        J_temp_ion1 = density_ion1 * vel_ion1 	# current density
        J_temp_ion1_err = density_ion1 * aver_err_ion1 
        
        e=1.602*10**(-19)
        J_C_nm2ns_ion1 = J_temp_ion1 * z_ion1 * e	
        J_C_nm2ns_ion1_err = J_temp_ion1_err * z_ion1 * e	
        
        # Unit to C/(m^2 s)
        J_ion1 = J_C_nm2ns_ion1 * (10**(9))**2 * 10**(9)
        J_ion1_err = J_C_nm2ns_ion1_err * (10**(9))**2 * 10**(9)
        
        print("Current density : ",J_ion1," [C/(m^2 s)]")
        print('J_ion1/self.Ex, ', J_ion1, self.Ex)
        
        sigma_ion1 = J_ion1/self.Ex * 10**(-9)
        sigma_ion1_err = J_ion1_err/self.Ex * 10**(-9)
        
        print("Conductivity_ion1 : ",sigma_ion1,"(+/- ", sigma_ion1_err,") [ampere^2 s^3/(kg m^3)]")
        return J_ion1, J_ion1_err, sigma_ion1, sigma_ion1_err


    def run(self, **kwargs):
#    def run(self, start, stop, step):
        def compute_stderror(s, sq, c):
            #s: sum; c: count
            return np.sqrt((sq/c) - (s*s)/(c*c) ) / (np.sqrt(c))

        def aver_err(arr,aver):
        	length_arr = len(arr)
        	sum_deviation = 0
        	for i in range(len(arr)):
        		sum_deviation = sum_deviation + (arr[i]-aver)**2
        	std_deviation = np.sqrt((1./(length_arr-1)) * sum_deviation)		# s= np.sqrt( sum_N (x_i - x^bar)**2 / N-1 )
        	return std_deviation / np.sqrt(length_arr)				# Std_error = s/np.sqrt(N)
        
        def err_prop(a,b):
        	return np.sqrt(a**2 + b**2)
        
        self._timeseries = []
        self.timesteps = []
        self._s1 = self.u.select_atoms(self.selection1)
        s1_ids=self._s1.ids
        s1_tot_res = len(self._s1.ids)
        self.s1_tot_res = s1_tot_res

        ####  Atom selection
        _s1 = self.u.select_atoms(self.selection1)
        _s2 = self.u.select_atoms(self.selection2)
     
        #### Collect atomics positions
        self.u.trajectory[0]
        t_first_ns = self.u.trajectory.time / 1000

        ion1_pos_first = np.array(_s1.positions) / 10   # Angstrom to nm
        ion2_pos_first = np.array(_s2.positions) / 10   # Angstrom to nm

        self.u.trajectory[-1]
        t_last_ns = self.u.trajectory.time / 1000

        _s1 = self.u.select_atoms(self.selection1)
        ion1_pos_last = np.array(_s1.positions) / 10    # Angstrom to nm
        _s2 = self.u.select_atoms(self.selection2)
        ion2_pos_last = np.array(_s2.positions) / 10   # Angstrom to nm
        
        ion1_av_vel =  ( ion1_pos_last - ion1_pos_first )  / (t_last_ns - t_first_ns)
        ion2_av_vel =  ( ion2_pos_last - ion2_pos_first )  / (t_last_ns - t_first_ns)

        ion1_av_vel_x = ion1_av_vel[:,0]
        ion2_av_vel_x = ion2_av_vel[:,0]

        ####  Compute density 
        n_ion1 = len(_s1)       #float(argv[5])
        n_ion2 = len(_s2)       #float(argv[6])

        n_tot = n_ion1 + n_ion2
        box_l= self.u.dimensions[0] / 10   #float(argv[4]) # For the density calculations [nm]
        print("Ex : ", self.Ex)
        print("box_l : ", box_l)
        print("n_ion1 : ", n_ion1,"+ n_ion2 : ",n_ion2,"= n_tot :",n_tot)
        density_ion1 = n_ion1/(self.u.dimensions[0] * self.u.dimensions[1] * self.u.dimensions[2] ) * 1000   # AA^3 -> nm^3
        density_ion2 = n_ion2/(self.u.dimensions[0] * self.u.dimensions[1] * self.u.dimensions[2] ) * 1000
        density_tot = n_tot/(self.u.dimensions[0] * self.u.dimensions[1] * self.u.dimensions[2] ) * 1000
        
        ####  Atomic charges
        #z_ion1 = +1.0
        #z_ion2 = -2.0
        z_ion1 = +1.0  #float(argv[7])
        z_ion2 = -1.0  #float(argv[8])
        print("z1: %s z2: %s" %(z_ion1, z_ion2)); #input('enter')
        ##################

        
        #### Compute current density and conductivity 
        print("-----------------ion1--------------------")
        J_ion1, J_ion1_err, sigma_ion1, sigma_ion1_err = self._compute(ion1_av_vel_x, density_ion1, z_ion1)

        
        print("----------------- ion2 --------------------")
        J_ion2, J_ion2_err, sigma_ion2, sigma_ion2_err = self._compute(ion2_av_vel_x, density_ion2, z_ion2)

        ##################################################################### 
        
        print("-----------------Total--------------------")
        J_tot = J_ion1 + J_ion2
        #print('J_tot: ', J_tot, J_ion1, J_ion2)
        J_tot_myunits = J_tot * 10**(-6)
        J_tot_err = err_prop(J_ion1_err, J_ion2_err)
        J_tot_err_myunits = err_prop(J_ion1_err, J_ion2_err) * 10**(-6)
        
        print("Total current density : ",J_tot_myunits, "(+/- ",J_tot_err_myunits,") * 10**(6) [A/(m^2)]")
        
        sigma_tot = sigma_ion1 + sigma_ion2
        sigma_tot_err = err_prop(sigma_ion1_err,sigma_ion2_err)
        print("Total conductivity : ",sigma_tot,"(+/- ",sigma_tot_err,") [ampere^2 s^3/(kg m^3)]")
        
        fout = open('total_current_density.dat','a')
        fout.write('%s %s %s\n' %(self.Ex*1000, J_tot_myunits, J_tot_err_myunits))
        fout.close()
        
        fout = open('total_conductivity.dat','a')
        fout.write('%s %s %s\n' %(self.Ex*1000, sigma_tot, sigma_tot_err))
        fout.close()
        
        #print("-----------------transference number--------------------")
        #def t_err(x,dx,y,dy):
        #        z = float(x) / float(y)
        #        return z * np.sqrt( (float(dx)/float(x))**2 + (float(dy)/float(y))**2)
        #
        ##t_ion1 = sigma_ion1 / sigma_tot 
        ##t_ion2 = sigma_ion2 / sigma_tot 
        #t_ion1 = sigma_ion1 / sigma_tot
        #t_ion1_err = t_err(sigma_ion1, sigma_ion1_err , sigma_tot, sigma_tot_err )
        #t_ion2 = sigma_ion2 / sigma_tot
        #t_ion2_err = t_err(sigma_ion2, sigma_ion2_err , sigma_tot, sigma_tot_err )
        #
        #
        #print("Transference number of ion1 : %s +/- %s " %(t_ion1, t_ion1_err))
        #print("Transference number of ion2 : %s +/- %s " %(t_ion2, t_ion2_err))


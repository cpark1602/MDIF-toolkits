# Copyright (c) 2026 Chanbum Park chanbum.park@theochem.ruhr-uni-bochum.de 
# Distributed under the terms of the GNU General Public License.

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import MDAnalysis as mda
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from MDAnalysis.lib import distances

import kirkwood_gk as kw_gk

# Setup warnings and logging filter configurations
warnings.filterwarnings(action='once')
logger = logging.getLogger('MDAnalysis.analysis.hbonds')

class Kirkwood_Gk:
    def __init__(self, universe, box,  
                 print_results_path, pbc, bin_size, dim,
                 selection1, selection2, cutoff_IF, cutoff_BULK,  
                 start=None, stop=None, step=None): 
        
        #self.WC_angle_list_global = WC_angle_list_global
        self.u = universe
        self.seconds1 = time.time()

        self.total_frame = len(self.u.trajectory)
        print("Total frame:", self.total_frame)
        self.box = box
        self.selection1 = selection1
        self.selection2 = selection2
       
        #self.cutoff_dist_O_H = cutoff_dist_O_H
        #self.cutoff_dist_donor_acceptor = cutoff_dist_donor_acceptor
        self.OH_dist_cutoff = 1.2    
        
        self.cutoff_IF = cutoff_IF         
        self.cutoff_BULK = cutoff_BULK     
        self.cutoff_NE = 31
        
        #self.angle = angle
        self.pbc = pbc and all(self.u.dimensions[:3])
        self.start = start
        self.stop = stop
        self.step = step
        
        self.print_results_path = print_results_path
        if not os.path.exists(self.print_results_path):
            os.makedirs(self.print_results_path)

    def _get_bonded_hydrogens_dist(self, atom):
        """Find bonded water hydrogens within strict structural cutoff limits to target Oxygen."""
        try:
            sel_h = atom.residue.atoms.select_atoms(
                "(name HW1 or name HW2) and around {0:f} index {1!s}".format(self.OH_dist_cutoff, atom.index))
            return sel_h
        except Exception:
            return []
        
    def _update_selection_1(self):
        """Dynamically updates unique configurations for Hydrogens linked to current Donor targets."""
        self._s1 = self.u.select_atoms(self.selection1)
        self._s1_donors_h = {}

        for i, d in enumerate(self._s1):
            tmp = self._get_bonded_hydrogens_dist(d)
            if tmp:
                self._s1_donors_h[i] = tmp

    def _run_window_i(self, start, stop, step):
        """Processes target micro-trajectory segment windows to compute system properties."""
        self._s1 = self.u.select_atoms(self.selection1)
        s1_tot_res = len(self._s1.ids)
        self.dict_Oatom_index_to_i = {}

        n_frames_i = 0
        self.u.dimensions = self.box

        logger.info("HBond analysis: starting window step loop")

        # Initializing local dependencies
        kw_gk_u = kw_gk.Kirkwood_Gk(self.u)

        for ts in self.u.trajectory[start:stop:step]:
            frame = ts.frame
            print(f"Frame: {frame:5d}\n-----------------------------")

            self.dict_oxygen_with_h_tagged = {}
            n_frames_i += 1
            
            self._s1 = self.u.select_atoms(self.selection1)
            self._s2 = self.u.select_atoms(self.selection2)

            for i_s1 in range(len(self._s1)):
                self.dict_Oatom_index_to_i[self._s1[i_s1].index] = i_s1 

            self._update_selection_1()
            self.ns_acceptors = AtomNeighborSearch(self._s2, self.box)

            # Map Spatial density positions along longitude limits
            locpres = self.u.coord.positions
            JJ = self._s1.indices
            rxo = locpres[JJ, 0]
            
            nr_water_in_if = len(np.where(rxo < self.cutoff_IF[1])[0])

            if nr_water_in_if > 2:
                self.Hatoms = self.u.select_atoms('name H')
                for h_tag in self.Hatoms:
                    oxygens_for_Htagging = self.ns_acceptors.search(h_tag, 5.0)   
                    dict_dist_o_h_tagged = {} 
                    
                    for o_near_h in oxygens_for_Htagging:
                        dist_o_h_tagged = distances.calc_bonds(h_tag.position, o_near_h.position, box=self.box)
                        dict_dist_o_h_tagged[o_near_h.index] = dist_o_h_tagged
                    
                    dict_dist_o_h_tagged_sorted = {k: v for k, v in sorted(dict_dist_o_h_tagged.items(), key=lambda item: item[1])}
                    closest_o_near_h_tagged_index = list(dict_dist_o_h_tagged_sorted.keys())[0]

                    if closest_o_near_h_tagged_index in self.dict_oxygen_with_h_tagged:
                        self.dict_oxygen_with_h_tagged[closest_o_near_h_tagged_index].append(h_tag)
                    else:
                        self.dict_oxygen_with_h_tagged[closest_o_near_h_tagged_index] = [h_tag]

                #for kw_gk_cutoff in [0.5, 1, 2, 2.9, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]: 
                for kw_gk_cutoff in [0.5, 1, 2]: 
                    kw_gk_u.run(ts, n_frames_i, self._s1, self.dict_Oatom_index_to_i, 
                                self.dict_oxygen_with_h_tagged, kw_gk_cutoff, self.box)

        print("=== End of ts window loop ===") 
        kw_gk_mu_aver_global = kw_gk_u.mu_aver / kw_gk_u.count  
        return kw_gk_mu_aver_global

    def _slice_trj(self):
        """Generates dynamic frame window slices mapped into chunks of 10,000 frames."""
        sampling_numbers = 100
        tmp = int((self.stop - self.start) / self.step)
        nruns = np.ceil(tmp / sampling_numbers)

        window_list = []
        windows = 0
        for i in range(int(nruns + 1)):
            window_list.append(windows)
            windows += sampling_numbers
        return window_list

    def run(self, **kwargs):
        """Main execution gateway handling loop window scheduling and data archiving."""
        window_list = self._slice_trj()
        nruns = len(window_list)
        print('Total window segments to process: ', nruns - 1)

        kw_gk_mu_aver_global = 0.0

        for i in range(nruns - 1):
            print('Processing window block: ', i)
            kw_gk_mu_aver = self._run_window_i(window_list[i], window_list[i+1], self.step) 
            kw_gk_mu_aver_global += kw_gk_mu_aver

        kw_gk_mu_aver_global /= (nruns - 1)

        #output_file = os.path.join(self.print_results_path, 'kw_gk_mu_aver_globa.npy')
        #np.save(output_file, kw_gk_mu_aver_global)

        self.seconds2 = time.time()
        run_time = self.seconds2 - self.seconds1
        print(f"Analysis complete. Total runtime: {run_time / 60.0:.2f} Minutes")
        return kw_gk_mu_aver_global

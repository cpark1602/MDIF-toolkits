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

import matplotlib.pyplot as pl
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import matplotlib.font_manager as fm
font_names = [f.name for f in fm.fontManager.ttflist]
# # Hydrogen bond analysis
import warnings
import logging
from MDAnalysis import MissingDataWarning, NoDataError, SelectionError, SelectionWarning
from MDAnalysis.lib.log import ProgressBar
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from MDAnalysis.lib import distances
from MDAnalysis.lib.correlations import autocorrelation, correct_intermittency
logger = logging.getLogger('MDAnalysis.analysis.hbonds')

class ACF:
    def __init__(self, universe, box, HBs_criteria, selection1, selection2, path_results, print_results_path, cutoff_dist_O_H, cutoff_dist_donor_acceptor, cutoff_IF, cutoff_BULK, angle, pbc=True, start=None, stop=None, step=None, nac='IF', **kwargs):   # 
        
        self.u = universe
        self.seconds1 = time.time()
        self.t_l = []

        self.total_frame = len(self.u.trajectory)
        print("Total frame:", self.total_frame)
        self.box = box
        self.selection1 = selection1
        self.selection2 = selection2
        self.n_frames = None
        
        # Cutoff for searching O/H atoms around the reference
        self.cutoff_dist_O_H = cutoff_dist_O_H
        # Cutoff for the distance between O-O.
        self.cutoff_dist_donor_acceptor = cutoff_dist_donor_acceptor
        # a bonded O-H distance cutoff; used in _gen_bonded_hydrogens_dist
        self.OH_dist_cutoff = 1.2    
        self.HBs_criteria = HBs_criteria
        
        # cutoff for the donor position; the values are based on the density profile Fig. 2.4 in the report
        self.cutoff_IF = cutoff_IF         #12
        self.cutoff_BULK = cutoff_BULK     #[19, 28] between 19 to 28 Angstrom 
        self.cutoff_NE = 31
        
        # Cutoff angle O-H-O
        self.angle = angle
        self.pbc = pbc and all(self.u.dimensions[:3])
        # Start frame nr. and stop frame nr.
        self.start = start
        self.stop = stop
        self.step= step
        self.search_val = re.compile('11.6355*')
        # final result accessed as self.timeseries
        self._timeseries = None  
        # time for each frame
        self.timesteps = None
        # The hydrogen bonds O-H-O list for pandas
        self.table = None
        self.timeseries_table = None
        
        print("path_results: ", path_results)
        self.path_results = path_results

        self.total_count_array_BULK_global = None
        self.count_array_IF_global_aver = None
        self.count_array_BULK_global_aver = None

        # N_donor/accepter
        self.count_array_IF_global_Ndonor_aver = None
        self.count_array_IF_global_Nacceptor_aver = None
        self.count_array_BULK_global_DAnumber_aver = None
        
        
        ### HBs ACF
        self.hb_acf_results = None

        self.print_results_path = print_results_path
        if not os.path.exists(self.print_results_path):
            os.makedirs(self.print_results_path)


    def _get_bonded_hydrogens_dist(self, atom):
        """Find bonded hydrogens within cutoff to 'atom'.
        Hydrogen bonds are detected by the cutoff;
        The distance from the reference 'atom' is calculated for all hydrogens in the residue
        and only those within a cutoff are kept."""
        try:
            sel_h = atom.residue.atoms.select_atoms(
                "(name HW1 or name HW2) and around {0:f} index {1!s}".format(self.OH_dist_cutoff, atom.index))
            return sel_h
            
        except NoDataError:
            return []
        
    def _update_selection_1(self):
        """Update the hydrogens around Oxygen atoms"""
        self._s1 = self.u.select_atoms(self.selection1)
        self._s1_donors = {}
        self._s1_donors_h = {}
        self._s1_acceptors = {}

        # d: donor atom_select
        for i, d in enumerate(self._s1):
            tmp = self._get_bonded_hydrogens_dist(d)
            #print('get bonded h: ', tmp)
            if tmp:
                self._s1_donors_h[i] = tmp   # fill the dict[i]

                
    def _update_selection_2(self):
        """Update the hydrogens around Oxygen atoms"""
        self._s2 = self.u.select_atoms(self.selection2)
        self._s2_donors = {}
        self._s2_donors_h = {}
        self._s2_acceptors = {}
        for i, d in enumerate(self._s2):
            tmp = self._get_bonded_hydrogens_dist(d)
            #print(tmp)
            if tmp:
                self._s2_donors_h[i] = tmp   # fill the dict[i]

#    def run(self, **kwargs):
    def _single_run(self, start, stop, step):
        
        self._timeseries = []
        self.timesteps = []
        self._s1 = self.u.select_atoms(self.selection1)
        s1_ids=self._s1.ids
        s1_tot_res = len(self._s1.ids)
        #print("Nr. of donors: ", s1_tot_res); input('enter')
        self.s1_tot_res = s1_tot_res
        
        count_array_IF_global = np.zeros((3,3), dtype=np.float64)
        count_array_IM_global = np.zeros((3,3), dtype=np.float64)
        count_array_BULK_global = np.zeros((4,4), dtype=np.float64)
        count_array_NE_global = np.zeros((3,3), dtype=np.float64)

        self.dict_Oatom_index_to_i = {}

        "----------HB ACF----------"
        #already_found_first_frame = {}                    # disc for the HB ACF
        #prev_already_found = {}                           # disc for the HB ACF
        #self.hb_acf_results = np.zeros_like(np.arange(self.start, self.stop, self.step), dtype=np.float32)
        
        already_found_first_frame_IF = {}                    # disc for the HB ACF
        prev_already_found_IF = {}                           # disc for the HB ACF
        
        already_found_first_frame_BULK = {}                    # disc for the HB ACF
        prev_already_found_BULK = {}                           # disc for the HB ACF

        hb_acf_results_IF = np.zeros_like(np.arange(start, stop, step), dtype=np.float32)
        hb_acf_results_BULK = np.zeros_like(np.arange(start, stop, step), dtype=np.float32)
        nr_HBs_IF_t = np.zeros_like(np.arange(start, stop, step), dtype=np.float32)
        nr_HBs_BULK_t = np.zeros_like(np.arange(start, stop, step), dtype=np.float32)
        nr_HBs_NE_t = np.zeros_like(np.arange(start, stop, step), dtype=np.float32)
        tot_hb = s1_tot_res * 2
        print('tot_hb', tot_hb)#;input('eter')
        "---------------------------"
        print('start acf: ', hb_acf_results_IF); #input('enter')

        self.dim_ndx = 0
        L = self.box[0]
        #print("n_bins: ", self.n_bins)
        #frames = 0
        # Initiate the histogram
        # use linspace instead of histogram
        #print("h3oDensity Hist edge len", len(h3oDensity_hist_edges))

        logger.info("HBond analysis: starting")

        n_frames_i = 0
        stepsize=0.5
        for ts in self.u.trajectory[start:stop:step]:
        #for ts in ProgressBar(self.u.trajectory[self.start:self.stop:self.step],
        #                      desc="HBond analysis",
        #                      verbose=kwargs.get('verbose', False)):
            # all bonds for this timestep
            frame_results = []
            # dict of tuples (atom.index, atom.index) for quick check if
            # we already have the bond (to avoid duplicates)
            #print("Frame: {0:5d}, Time: {1:8.3f} ps".format(ts.frame, self.u.trajectory.time))
            print("Frame: {0:5d}".format(ts.frame))

            already_found = {}
            "----------HB ACF----------"
            #hb_acf_already_found = {}
            hb_acf_already_found_IF = {}
            hb_acf_already_found_BULK = {}
            "--------------------------"
            dict_d_a = {}
            dict_a_d = {}
       
            # Net atomic charge  
            count_array_IF_charge_H = np.zeros((3,3), dtype=np.float64)
            count_array_IM_charge_H = np.zeros((3,3), dtype=np.float64)
            count_array_BULK_charge_H = np.zeros((4,4), dtype=np.float64)
            count_array_NE_charge_H = np.zeros((3,3), dtype=np.float64)

            count_array_IF_charge_H_count = np.zeros((3,3), dtype=np.float64)
            count_array_IM_charge_H_count = np.zeros((3,3), dtype=np.float64)
            count_array_BULK_charge_H_count = np.zeros((4,4), dtype=np.float64)
            count_array_NE_charge_H_count = np.zeros((3,3), dtype=np.float64)

            count_array_IF_charge_O = np.zeros((3,3), dtype=np.float64)
            count_array_IM_charge_O = np.zeros((3,3), dtype=np.float64)
            count_array_BULK_charge_O = np.zeros((4,4), dtype=np.float64)
            count_array_NE_charge_O = np.zeros((3,3), dtype=np.float64)

            count_array_IF_charge_O_count = np.zeros((3,3), dtype=np.float64)
            count_array_IM_charge_O_count = np.zeros((3,3), dtype=np.float64)
            count_array_BULK_charge_O_count = np.zeros((4,4), dtype=np.float64)
            count_array_NE_charge_O_count = np.zeros((3,3), dtype=np.float64)

            #
            count_array_IF = np.zeros((3,3), dtype=np.float64)
            count_array_IF_Ndonor = np.zeros((3,3), dtype=np.float64)
            count_array_IF_Nacceptor = np.zeros((3,3), dtype=np.float64)

            self.count_array_IF_EXT = np.zeros((6,6), dtype=np.float64)       # Extended HB matrix
            self.count_array_IF_EXT_global = np.zeros((6,6), dtype=np.float64)       # Extended HB matrix

            count_array_IM = np.zeros((3,3), dtype=np.float64)
            count_array_BULK = np.zeros((4,4), dtype=np.float64)
            count_array_NE = np.zeros((3,3), dtype=np.float64)

            self.oxygen_pos_angle_xaxis_HB_tuple_IF = {} 
            self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF = {}
            self.hydrogen_pos_angle_xaxis_HB_tuple_IF = {} 
            self.oxygen_pos_netWaterchg = {}

            self.hist_tuple_angle_charge_IF_O = []
            self.hist_tuple_angle_charge_IF_H = []

            self.HB_collect_d1ax = []
            self.HB_collect_d2ax = []

            "Dictionary for oxygen atoms with H tagged"
            self.dict_oxygen_with_h_tagged = {}

            frame = ts.frame
            print("-----------------------------")
            #print('time steps: ', self.timesteps)
            #print("frame: %s"%frame)
            #if frame == self.start:
            #    print("")
            #    print("************************")
            #    print("ts.frame == frame: %s %s" %(frame, self.start))
            #    print("")

            #print("")
            #print("--------------------------------")           
            #print("Frame: {0:5d}, Time: {1:8.3f} ps".format(ts.frame, self.u.trajectory.time)); input('enter')

            n_frames_i += 1
            time_fs = ts.frame * stepsize   # * self.step; ts.frame already include the step 
            
            #self.timesteps.append(self.u.trajectory.time)
            self.timesteps.append(time_fs)

            #n_frableames += 1
            print('n_frames: ', n_frames_i)
            # Select atom groups
            self._s1 = self.u.select_atoms(self.selection1)
            self._s2 = self.u.select_atoms(self.selection2)

            for i_s1 in range(len(self._s1)):
                self.dict_Oatom_index_to_i[self._s1[i_s1].index] = i_s1 
                #print("i: %s, index: %s" %(i_s1, self._s1[i_s1].index))

            for i_s1 in range(len(self._s1)):
                ndx_s1 = self._s1[i_s1].index
                #i_s1 = self.index_to_i_dict(ndx_s1)
                #print("i: %s, index: %s," %(i_s1, self.dict_Oatom_index_to_i[ndx_s1]))
            
            # Update the HB list
            #if self.update_selection1:
            """_update_selection_1() call the function _get_bonded_hydrogens_dist"""
            """to find the bonded Hydrogens"""
            self._update_selection_1()
            #if self.update_selection2:
            self._update_selection_2()

            # Call the class AtomNeighborSearch wrt _s2
            # Return all atoms/residues/segments that are within *radius* of the atoms in *atoms*.
            # ns_acceptor is a module, which is called later to search within 'cutoff'.** 
            self.ns_acceptors = AtomNeighborSearch(self._s2, self.box)

            # Count number water molecules in IF/BULK/Ne
            locpres= self.u.coord.positions
            JJ=self._s1.indices
            rxo=locpres[JJ,0]
            #print(self.cutoff_IF); input('exit')
            nr_water_in_if = np.where(rxo < self.cutoff_IF[1])
            nr_water_in_if = len(nr_water_in_if[0])
            nr_HBs_IF_t[n_frames_i-1] = nr_water_in_if * 2

            nr_water_in_bulk = np.where( (rxo > self.cutoff_BULK[0]) & ( rxo < self.cutoff_BULK[1] ) )
            nr_water_in_bulk = len(nr_water_in_bulk[0])
            nr_HBs_BULK_t[n_frames_i-1] = nr_water_in_bulk * 2

            nr_water_in_ne = np.where(rxo > self.cutoff_NE)
            nr_water_in_ne = len(nr_water_in_ne[0])
            nr_HBs_NE_t[n_frames_i-1] = nr_water_in_ne * 2


            h3o_activeHbond_hydrogen = []
           
            """ search oxygens, which are closest hydrogen atoms  """
            self.Hatoms = self.u.select_atoms('name H')
            #print('Hatoms: ', len(self.Hatoms))
            for h_tag in self.Hatoms:
                #h_tag = self.Hatoms[0]
                #print('h_tag index: ', h_tag.index, h_tag.position); input('enter')
                oxygens_for_Htagging = self.ns_acceptors.search(h_tag, 5.0)   # self.box[0]*2 self.box[0], search oxygens within the simulation box
                #print(oxygens_for_Htagging[0].index)
                dict_dist_o_h_tagged = {} 
                ### Search oxygen near the reference hydrogen within the cutoff
                for o_near_h in oxygens_for_Htagging:
                    dist_o_h_tagged = distances.calc_bonds(h_tag.position, o_near_h.position, box=self.box)
                    #print("index O near h_i: %s, h_index: %s, dist: %s" %( o_near_h.index, h_tag.index, dist_o_h_tagged) )
                    dict_dist_o_h_tagged[o_near_h.index] = dist_o_h_tagged
                ### sort the dictionary to find the closest oxygen
                dict_dist_o_h_tagged_sorted = {k: v for k, v in sorted(dict_dist_o_h_tagged.items(), key=lambda item: item[1])}
                #print(dict_dist_o_h_tagged_sorted)
                closest_o_near_h_tagged_index =  list(dict_dist_o_h_tagged_sorted.keys())[0]

                if closest_o_near_h_tagged_index in self.dict_oxygen_with_h_tagged:
                    #print('key exist')
                    self.dict_oxygen_with_h_tagged[closest_o_near_h_tagged_index].append(h_tag)
                    #print('added')
                    #print(self.dict_oxygen_with_h_tagged)
                else:
                    #print('key does not exist')
                    #print(self.dict_oxygen_with_h_tagged)
                    self.dict_oxygen_with_h_tagged[closest_o_near_h_tagged_index] = [h_tag]
                    #print('added')
                    #print(self.dict_oxygen_with_h_tagged)
            ###=== Tagging H finished===###

            #count_o_h_tagged = 0
            #for o_tag in self._s1:
            #    print('count %s, o agindex %s' %( count_o_h_tagged, o_tag.index))
            #    count_o_h_tagged += 1

            ###--- loop over the oxygen dictionary with tagged H---###
            ###--- Oxygen atoms belong to ClO4 are not selected; because those are not the closest oxygens---###
            count_h3o = 0
            h3o_dist_list = []; h3o_dist_da_dict = {}
            for key, value in self.dict_oxygen_with_h_tagged.items():
                ###key: donor index, values: the closest hydrogen atoms 
                #print("loop over the oxygen index: ", key)
                #d = self._s1[self.index_to_i(key)]   # index_to_i is wrong; becaue oh- is made by removeing two h from h3o-> The sequence is wrong.
                d = self._s1[self.dict_Oatom_index_to_i[key]]   # use a new dictionary which is sorted according to the self._s1 list.

                #donor_pos_xyz = self._s1[self.index_to_i(key)].position
                #donor_index = self._s1[self.index_to_i(key)].index
                donor_pos_xyz = d.position
                donor_index = d.index
                
                # loop over tagged H, which belongs to the O_i
                for h in value:               
                    ### Hydrogen density profile.
                    #HydrogenDensity_hist_ndx = np.digitize(h.position[self.dim_ndx], bins=h3oDensity_hist_edges)
                    #HydrogenDensity_hist[HydrogenDensity_hist_ndx] += 1

                    res = self.ns_acceptors.search(h, self.cutoff_dist_O_H)
                    # d-h -- a ; Search a around h within cutoff_dist_O_H 

                    for a in res:
                        #print(donor_index, a.index) 

                        if d.index == a.index:
                            #print("duplicated pairs donor %s acceptor %s" %(d.index, a.index))
                            #print("skip...\n")
                            #continue
                            pass
                        else: 
                            #print("Main loop: indexes of the d-h-a: %s-%s-%s " %(d.index, h.index, a.index))

#                            if (d.index, h.index, a.index) in already_found or (a.index, h.index, d.index) in already_found:
                            if (d.index, h.index, a.index) in already_found:
                                #if d.index == 167: 
                                # 	print("O-H-O %s-%s-%s already found, skip. ***" %(a.index, h.index, d.index))
                                #continue
                                pass
                            else:
                                #print("--- in loop ---")
                                angle_rad = distances.calc_angles(h.position, 
                                                              d.position,
                                                              a.position, box=self.box)
                                
                                angle_OHO_rad = distances.calc_angles(d.position, 
                                                              h.position,
                                                              a.position, box=self.box)
                                
                                angle = np.rad2deg(angle_rad)
                                dist = distances.calc_bonds(d.position, a.position, box=self.box)
                                
                                dist_O1_H = distances.calc_bonds(d.position, h.position, box=self.box)
                                dist_O2_H = distances.calc_bonds(a.position, h.position, box=self.box)

                                "Rectangular HBs cutoffs"
                                if self.HBs_criteria == 'Luzar':
                                    #print("Luzar")
                                    if angle <= self.angle and dist <= self.cutoff_dist_donor_acceptor:
                                        #print("****************** hydrogen position: %s\n  Donor position: %s\n, acceptor position: %s" %(h.position, d.position, a.position))
                                    
                                        #print("****************** HB Found: %s-%s-%s; dist:%s angle: %s" %(d.index, h.index, a.index, dist, angle))
                                        #print("****************** location: ", h.position[0],d.position[0], a.position[0])
                                        #frame_results.append( [d.index, h.index, a.index, (d.name, h.name, a.name), h.position[0], dist, angle])
                                        already_found[(d.index, h.index, a.index)] = True
                                               
                                        # Make a new dict for analysis
                                        # Add the dornor and acceptor pair into the dict.
                                        # Add new value to the list of the key if a key does not exist, make a new key
                                        dict_d_a.setdefault(d,[]).append(a)
                                        dict_a_d.setdefault(a.index,[]).append(d)
                                        #c += 1   

                                        # Find active H in the H3O+, update the histograme of the free energy of H3O+
                                        if len(value) == 3: 
                                            dist_1 = distances.calc_bonds(h.position, d.position, box=self.box)
                                            dist_2 = distances.calc_bonds(h.position, a.position, box=self.box)
                                            dist_delta_d_a = abs(dist_1 - dist_2)
                                        #    #print(' d(%s)-h(%s)-a(%s): dist delta: %.4f' %(d.index, h.index, a.index, dist_delta_d_a))
                                            h3o_dist_list.append(dist_delta_d_a) 
                                            h3o_dist_da_dict[a.index] = dist_delta_d_a 
                                            h3o_activeHbond_donor = d.index
                                        #    #print('  Add all keys and values in continous list, len(%s)\n %s' %( len(dict_d_a_delta_continuous), dict_d_a_delta_continuous) )

                                        "#####***** HBs ACF (IF) *****#####"
                                        if self.cutoff_IF[0] < d.position[0] <= self.cutoff_IF[1]:
                                            if frame == start:
                                                prev_already_found_IF[(d.index, h.index, a.index)] = True
                                            else:
                                                #print("----- NOT first frame-----")
                                                # search the same d-h-a, which survive from the previous 
                                                if (d.index, h.index, a.index) in prev_already_found_IF:
                                                    hb_acf_already_found_IF[(d.index, h.index, a.index)] = True
                                                    hb_acf_results_IF[n_frames_i-1] += 1

                                        "#####***** HBs ACF (BULK) *****#####"
                                        if d.position[0] >= self.cutoff_BULK[0] and d.position[0] <= self.cutoff_BULK[1]:
                                            if frame == start:
                                                prev_already_found_BULK[(d.index, h.index, a.index)] = True
                                            else:
                                                if (d.index, h.index, a.index) in prev_already_found_BULK:
                                                    hb_acf_already_found_BULK[(d.index, h.index, a.index)] = True
                                                    hb_acf_results_BULK[n_frames_i-1] += 1


                                #"Triangular HBs cutoffs"    
                                elif self.HBs_criteria == 'Sho':
                                    #print("Sho")
                                    cosine_term = -1.71 * np.cos(angle_OHO_rad) + 1.37
                                    #print("dist O_H: ", dist_O2_H, "cosine_term: ", cosine_term)
                                    if dist_O2_H < cosine_term:
                                        #print("****************** HB (Sho) Found: dist h-o(a) %s cos %s" %(dist_O2_H, cosine_term))
                                        #print("****************** location: ", h.position[0],d.position[0], a.position[0])
                                        #frame_results.append( [d.index, h.index, a.index, (d.name, h.name, a.name), h.position[0], dist, angle])
                                        already_found[(d.index, h.index, a.index)] = True
                                    
                                        # Make a new dict for analysis
                                        # Add the dornor and acceptor pair into the dict.
                                        # Add new value to the list of the key if a key does not exist, make a new key
                                        dict_d_a.setdefault(d,[]).append(a)
                                        dict_a_d.setdefault(a.index,[]).append(d)

                                        # Find active H in the H3O+, update the histograme of the free energy of H3O+
                                        if len(value) == 3: 
                                            dist_1 = distances.calc_bonds(h.position, d.position, box=self.box)
                                            dist_2 = distances.calc_bonds(h.position, a.position, box=self.box)
                                            dist_delta_d_a = abs(dist_1 - dist_2)
                                            #print(' d(%s)-h(%s)-a(%s): dist delta: %.4f' %(d.index, h.index, a.index, dist_delta_d_a))
                                            h3o_dist_list.append(dist_delta_d_a) 
                                            h3o_dist_da_dict[a.index] = dist_delta_d_a 
                                            h3o_activeHbond_donor = d.index
                                        #    #print('  Add all keys and values in continous list, len(%s)\n %s' %( len(dict_d_a_delta_continuous), dict_d_a_delta_continuous) )

                                        "#####***** HBs ACF (IF) *****#####"
                                        if self.cutoff_IF[0] < d.position[0] <= self.cutoff_IF[1]:
                                            if frame == start:
                                                prev_already_found_IF[(d.index, h.index, a.index)] = True
                                            else:
                                                #print("----- NOT first frame-----")
                                                # search the same d-h-a, which survive from the previous 
                                                if (d.index, h.index, a.index) in prev_already_found_IF:
                                                    hb_acf_already_found_IF[(d.index, h.index, a.index)] = True
                                                    hb_acf_results_IF[n_frames_i-1] += 1

                                        "#####***** HBs ACF (BULK) *****#####"
                                        if d.position[0] >= self.cutoff_BULK[0] and d.position[0] <= self.cutoff_BULK[1]:
                                            if frame == start:
                                                prev_already_found_BULK[(d.index, h.index, a.index)] = True
                                            else:
                                                if (d.index, h.index, a.index) in prev_already_found_BULK:
                                                    hb_acf_already_found_BULK[(d.index, h.index, a.index)] = True
                                                    hb_acf_results_BULK[n_frames_i-1] += 1



            "###********** HB ACF (IF) ***********###"
            if frame == start:
                already_found_first_frame_IF = prev_already_found_IF
                len_hb_acf_firstFrame_IF = len(already_found_first_frame_IF)
                hb_acf_results_IF[n_frames_i-1] = len_hb_acf_firstFrame_IF 
                #prev_already_found_IF = already_found_IF
                #print("first frame", prev_already_found_IF, len_hb_acf_firstFrame_IF)
            else:
                #if prev_already_found_IF != hb_acf_already_found_IF:
                    #print("not first frame ", prev_already_found_IF, len(prev_already_found_IF) )
                    #print("        current ", hb_acf_already_found_IF, len(hb_acf_already_found_IF) )
                    #input('enter')
                prev_already_found_IF = hb_acf_already_found_IF 
            
                
            "###********** HB ACF (BULK) ***********###"
            if frame == start:
                already_found_first_frame_BULK = prev_already_found_BULK
                len_hb_acf_firstFrame_BULK = len(already_found_first_frame_BULK)
                hb_acf_results_BULK[n_frames_i-1] = len_hb_acf_firstFrame_BULK
                #prev_already_found_IF = already_found_IF
            else:
                prev_already_found_BULK = hb_acf_already_found_BULK
                
            ##"###********** HB ACF ***********###"
            ##if frame == start:
            ##    already_found_first_frame = already_found
            ##    len_hb_acf_firstFrame = len(already_found_first_frame)
            ##    self.hb_acf_results[n_frames_i-1] = len_hb_acf_firstFrame 
            ##    prev_already_found = already_found
            ##else:
            ##    prev_already_found = hb_acf_already_found
            #
            #print(already_found_first_frame_IF)
            ##else:
            ##    self.hb_acf_results[n_frames_i] = self.hb_acf_results[n_frames_i] / len_hb_acf_firstFrame
            ##print("----------------")   # (ff)
            ##print("len HB_ACF: {}".format(len(hb_acf_already_found)))
            ##print("HB ACF results: %s" %(self.hb_acf_results))
            ##print("----------------")
            #
            #t2=time.time()
            #dt=t2-t1
            #self.t_l.append(dt)

        print("=== End of ts loop ===") 

        def compute_stderror(s, sq, c):
            #s: sum; c: count
            return np.sqrt((sq/c) - (s*s)/(c*c) ) / (np.sqrt(c))


        #print(n_frames_i, stepsize, self.step)

        #print('time fs: ', time_fs)
            
        #"---------- HB ACF ----------"
        # Normalzed by the first frame 
        #hb_acf_results_IF /= len_hb_acf_firstFrame_IF
        #hb_acf_results_BULK /= len_hb_acf_firstFrame_BULK
        # Not normalze by the first frame 
        #print(hb_acf_results_IF, hb_acf_results_BULK)

        # Normalized by the total number of HB in each region.
        hb_acf_results_IF /= nr_HBs_IF_t   # h at if
        hb_acf_results_BULK /= nr_HBs_BULK_t  # h at bulk 

        c_t = np.array(self.timesteps)
        h_0_if = hb_acf_results_IF[0] 
        h_t_if = hb_acf_results_IF
        c_dot_if = - (( h_t_if - h_0_if ) / c_t ) * ( 1 - h_t_if )
        c_dot_if_2 = - (( h_t_if[1] - h_0_if ) / (c_t[1] - c_t[0]) ) * ( 1 - h_t_if )

        h_0_bulk = hb_acf_results_BULK[0] 
        h_t_bulk = hb_acf_results_BULK
        c_dot_bulk = - (( h_t_bulk - h_0_bulk ) / c_t ) * ( 1 - h_t_bulk )
        c_dot_bulk_2 = - (( h_t_bulk[1] - h_0_bulk ) / (c_t[1] - c_t[0]) ) * ( 1 - h_t_bulk )

        #print(h_t_if);input('enter')
        #print('\nh_0_if', h_0_if);
        #print('\nh_t_if', h_t_if);
        #print('\nc_dot_if', c_dot_if, c_dot_if_2); input('enter')
         
        self.seconds2 = time.time()
        run_time = self.seconds2 - self.seconds1
        runt_time = run_time/60
        #print("total run time: ", run_time, "Minutes")

        return hb_acf_results_IF, hb_acf_results_BULK, c_dot_if, c_dot_if_2


    def _slice_trj(self, nruns):
        #window = int(self.total_frame / nruns)
        #window = int(self.stop - self.start / nruns)
        print('n runs: ', nruns, 'window: ', window)
        window_list = []; windows = 0
        for i in range(nruns+1):
           window_list.append(windows)
           windows += window
        print('window_list: ', window_list)
        return window_list 

    def _slice_trj_fixed_window10000(self):
        #window = 10000
        #sampling_numbers = 1000
        sampling_numbers = 10000
        #sampling_numbers = 3

        #nruns = int((self.stop - self.start) / sampling_numbers / self.step) # old not good

        tmp = int((self.stop - self.start) / self.step)
        nruns = np.ceil(tmp / sampling_numbers)
        print('nruns: ', nruns)#; input('enter')
        #nruns = tmp / window

        #if tmp < window:
            #raise ValueError('The total frame is less than the minimum window %s\n EXIT' %(min_window))
        #else:
             
        #print('n runs: ', nruns, 'window: ', window)
        window_list = []; windows = 0
        for i in range(int(nruns+1)):
           window_list.append(windows)
           windows += sampling_numbers
        #print('window_list: ', window_list); input('enter')
        #input('enter') 
        return window_list

    def run(self, **kwargs):
        #self._single_run(self.start, self.stop, self.step) 
        step=1; 
        #window_list = self._slice_trj(nruns)    # [start, ..., intermediates..., end]
        window_list = self._slice_trj_fixed_window10000()
        #window_list = [0, 4000, 8000, 12000]       # [start,stop-start,stop]
        nruns=len(window_list);print('nruns: ', nruns-1)#; input('enter')

        # Prepare
        print(window_list)
        hb_acf_results_IF_global = np.zeros_like(np.arange(window_list[0], window_list[1], self.step), dtype=np.float32)
        hb_acf_results_IF_acf_global = np.zeros_like(np.arange(window_list[0], window_list[1], self.step), dtype=np.float32)

        hb_acf_results_BULK_global = np.zeros_like(np.arange(window_list[0], window_list[1], self.step), dtype=np.float32)
        hb_acf_results_BULK_acf_global = np.zeros_like(np.arange(window_list[0], window_list[1], self.step), dtype=np.float32)

        c_dot_if_global = np.zeros_like(np.arange(window_list[0], window_list[1], self.step), dtype=np.float32)
        c_dot_if_2_global = np.zeros_like(np.arange(window_list[0], window_list[1], self.step), dtype=np.float32)

        #print(np.shape(hb_acf_results_IF_global)); input('enter')

        # Average HB number at each starting point; Average of HB.
        hb_acf_results_IF_global = 0
        hb_acf_results_BULK_global = 0

        #hb_acf_results_IF_global = 0; hb_acf_results_BULK_global = 0; c_dot_if_global = 0
        for i in range(nruns-1):
            print('nth window: ', i)
            hb_acf_results_IF, hb_acf_results_BULK, c_dot_if, c_dot_if_2 = self._single_run(window_list[i], window_list[i+1], self.step) 
            c_dot_if_global += c_dot_if
            c_dot_if_2_global += c_dot_if_2

            # ACF <h(0)h(t)>
            acf_if = hb_acf_results_IF[0] * hb_acf_results_IF
            #print('acf_if', acf_if, len(acf_if), np.shape(hb_acf_results_IF_acf_global));input('enter')
            hb_acf_results_IF_acf_global += acf_if
            # To compute average h
            hb_acf_results_IF_global +=  hb_acf_results_IF[0]

            # ACF <h(0)h(t)>
            acf_bulk = hb_acf_results_BULK[0] * hb_acf_results_BULK
            hb_acf_results_BULK_acf_global += acf_bulk
            # To compute average h
            hb_acf_results_BULK_global +=  hb_acf_results_BULK[0]

            #print(hb_acf_results_IF_global, hb_acf_results_IF) 

        hb_acf_results_IF_acf_global = hb_acf_results_IF_acf_global / (nruns -1)
        hb_acf_results_BULK_acf_global = hb_acf_results_BULK_acf_global / (nruns -1)

        hb_acf_results_IF_global = hb_acf_results_IF_global / (nruns -1)
        hb_acf_results_BULK_global = hb_acf_results_BULK_global / (nruns -1)

        print('hb_acf_results_IF_acf_global ', hb_acf_results_IF_acf_global)
        print('hb_acf_results_BULK_acf_global', hb_acf_results_BULK_global)

        # Final ACF; <h(0)*h(t)> / <h>
        hb_C_IF = hb_acf_results_IF_acf_global / hb_acf_results_IF_global 
        hb_C_BULK = hb_acf_results_BULK_acf_global / hb_acf_results_BULK_global 

        c_dot_if_global = ( c_dot_if_global / (nruns -1) ) / hb_acf_results_IF_global
        c_dot_if_2_global = ( c_dot_if_2_global / (nruns -1) ) / hb_acf_results_IF_global


        np.save(self.print_results_path+'/hb_acf_results_IF.npy', hb_C_IF)
        np.save(self.print_results_path+'/hb_acf_results_BULK.npy', hb_C_BULK)
        np.save(self.print_results_path+'/c_dot_if_global.npy', c_dot_if_global)
        np.save(self.print_results_path+'/c_dot_if_2_global.npy', c_dot_if_2_global)


######----- Trj path -----
#w_path="./"
##from sys import argv
## Use only xyz format.
##trj_file1=argv[1]
#trj_file1=test-pos-pos-frc.xyz
#u_if_q0_nac = mda.Universe(trj_file1)
#
#
#print("total nr. of frame: ", len(u_if_q0_nac.trajectory))
#tot_frames = len(u_if_q0_nac.trajectory)
#boxX = 48.57; boxY = 15.667; boxZ = 15.076
##boxX = 23.0000; boxY = 23.0000; boxZ = 22.3404
#box = [boxX, boxY, boxZ, 90, 90, 90]
#"Velesco angle H-O-O 35 degrees"
#HBs_criteria_input = 'Sho'   # Luzar: a rectangule; Sho: Triangle
##start_stop_step = [0, tot_frames, 1]  # q0.0-region2-new/ddec
#start_stop_step = [0, 10000, 1]  # q0.0-region2-new/ddec
##start_stop_step = [0, 3, 1]  # q0.0-region2-new/ddec
#
#chemisorbed_cutoff_O = 10.5
#chemisorbed_cutoff_H = 9.85
#path_results='./results/atomic_charge/'   # where the NAC results are stored
#print_results_path=w_path+'/results-acf-cdot-xyz/'     # To save the results
#if_q0_nac = HBanalysis(u_if_q0_nac, box, HBs_criteria_input, 'name O', 'name O', path_results, print_results_path, cutoff_dist_O_H =3.5, cutoff_dist_donor_acceptor = 3.5, 
#            cutoff_IF = [0, 12], cutoff_BULK = [19, 28], angle=35.0, start=start_stop_step[0], stop=start_stop_step[1], step=start_stop_step[2], nac='IF')  #tot_frames
#if_q0_nac.run()
#
#print('Use xyz format')
#print('Ne cutoff for the HB ACF analysis is 31 AA')
#

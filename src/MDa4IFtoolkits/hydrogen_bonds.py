import os
import numpy as np
from MDAnalysis import MissingDataWarning, NoDataError, SelectionError, SelectionWarning
from MDAnalysis.lib import distances

class Hydrogen_bonds:
    def __init__(self, universe, box, print_results_path, pbc, bin_size, dim, selection1, cutoff_IF, cutoff_BULK, start=None, stop=None, step=None): 

        self.bin_size = bin_size
        self.u = universe
        self.pbc = pbc
        self.bin_size = bin_size
        self.dim = dim

        self.print_results_path = print_results_path
        if not os.path.exists(self.print_results_path):
            os.makedirs(self.print_results_path)

        self.box = box
        self.OH_dist_cutoff = 1.2    
        self.cutoff_IF = cutoff_IF         
        self.cutoff_BULK = cutoff_BULK     

        self.pbc = pbc and all(self.u.dimensions[:3])
        self.start = start
        self.stop = stop
        self.step= step
        self.start_end_skip = [start, stop, step]

        self.selection1 = selection1



    def _get_bonded_hydrogens_dist(self, atom):
        """Find bonded hydrogens within cutoff to 'atom'.
        Hydrogen bonds are detected by the cutoff;
        The distance from the reference 'atom' is calculated for all hydrogens in the residue
        and only those within a cutoff are kept."""
        try:
            sel_h = atom.residue.atoms.select_atoms(
                "(name H) and around {0:f} index {1!s}".format(self.OH_dist_cutoff, atom.index))
            return sel_h
            
        except NoDataError:
            return []


    def _update_selection_1(self):
        """Update the hydrogens around Oxygen atoms"""
        self._s1 = self.u.select_atoms(self.selection1)
        self._s1_donors = {}
        self._s1_donors_h = {}
        self._s1_acceptors = {}
        for i, d in enumerate(self._s1):
            tmp = self._get_bonded_hydrogens_dist(d)
            if tmp:
                self._s1_donors_h[i] = tmp   # fill the dict[i]


    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        
        cosTheta = v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        
        return cosTheta
    
    
    # Note: returns angle in degree and cosine(theta)
    def vector2degree(self, vec, axis): 
        if axis == 'x':
            axis_ndx = 0
            unit_vec = np.array([1, 0, 0])
        elif axis == 'y':
            axis_ndx = 1
            unit_vec = np.array([0, 1, 0])
        else:
            axis_ndx = 2
            unit_vec = np.array([0, 0, 1])
        
        cosTheta_inRadian = self.angle_between(vec, unit_vec)
        degree = np.rad2deg(np.arccos(cosTheta_inRadian))

        return degree, cosTheta_inRadian
    
    def _getCosTheta(self, ndx):
        d = self._s1[ndx]
        
        dcoordO = []; xcoordH=[]; ycoordH = []; zcoordH = [];  tmp_H_position = []; h_index_tmp_hb = []; 
        ohb = 0; 
        for i in range(len(self._s1_donors_h[ndx])):
            new_H_position = self._s1_donors_h[ndx][i].position[:3]
            h_index_tmp_hb.append(self._s1_donors_h[ndx][i].index)
            xcoordH.append(self._s1_donors_h[ndx][i].position[0])
            ycoordH.append(self._s1_donors_h[ndx][i].position[1])
            zcoordH.append(self._s1_donors_h[ndx][i].position[2])

            ohbond_dist = distances.calc_bonds(d.position, self._s1_donors_h[ndx][i].position, box=None) # self.box
            if ohbond_dist > 2:
                ohb = 1
        
        coordO = d.position 
        # no change in the x-direction
        dcoordO.append(float(0.0))
        bcell = self.box[1]; ccell = self.box[2]
        # check the y direction 
        if (coordO[1] >= bcell):
           dcoordO.append(m.floor(coordO[1]/bcell)*bcell)
        elif (coordO[1] < float(0)):
           dcoordO.append(-m.ceil(abs(coordO[1])/bcell)*bcell)
        else:
           dcoordO.append(float(0.0))
        # check the z direction 
        if (coordO[2] > ccell):
           dcoordO.append(m.floor(coordO[2]/ccell)*ccell)
        elif (coordO[2] <= float(0)):
           dcoordO.append(-m.ceil(abs(coordO[2])/ccell)*ccell)
        else:
           dcoordO.append(float(0.0))

        xcoordHnew = xcoordH[0]-dcoordO[0]
        ycoordHnew = ycoordH[0]-dcoordO[1]
        zcoordHnew = zcoordH[0]-dcoordO[2]
        tmp_H_position.append(np.array([xcoordHnew, ycoordHnew, zcoordHnew]))

        string1 = "  " + "H" + " " + str('{:20.10f}'.format(xcoordHnew)) + str('{:20.10f}'.format(ycoordHnew)) + str('{:20.10f}'.format(zcoordHnew)) + "\n"

        xcoordHnew = xcoordH[1]-dcoordO[0]
        ycoordHnew = ycoordH[1]-dcoordO[1]
        zcoordHnew = zcoordH[1]-dcoordO[2]
        #print('new H2 pos: ', xcoordHnew, ycoordHnew, zcoordHnew)
        tmp_H_position.append(np.array([xcoordHnew, ycoordHnew, zcoordHnew]))

        string2 = "  " + "H" + " " + str('{:20.10f}'.format(xcoordHnew)) + str('{:20.10f}'.format(ycoordHnew)) + str('{:20.10f}'.format(zcoordHnew)) + "\n"

        coordO[:] = [float(p-q) for (p,q) in zip(coordO,dcoordO)]
        string3 = "  " + "O" + " " + str('{:20.10f}'.format(coordO[0])) + str('{:20.10f}'.format(coordO[1])) + str('{:20.10f}'.format(coordO[2])) + "\n"

        if ohb == 1: 
            for hb_iii in range(len(tmp_H_position)):
                dcoordH_a = []
                ohbond_dist = distances.calc_bonds(coordO, tmp_H_position[hb_iii], box=None) # self.box
                if ohbond_dist > 2.0:
                    #print('ohbond_dist', ohbond_dist, tmp_H_position[hb_iii])
                    ohbond_dist_pbc = distances.calc_bonds(coordO, tmp_H_position[hb_iii], box=self.box) # 
                    dcoordH_a.append(float(0.0))       
                    bcell = self.box[1]; ccell = self.box[2]
                    # check the y direction 
                    if (tmp_H_position[hb_iii][1] >= bcell):
                       dcoordH_a.append(m.floor(tmp_H_position[hb_iii][1]/bcell)*bcell)
                    elif (tmp_H_position[hb_iii][1] < float(0)):
                       dcoordH_a.append(-m.ceil(abs(tmp_H_position[hb_iii][1])/bcell)*bcell)
                    else:
                       dcoordH_a.append(float(0.0))
                    # check the z direction 
                    if (tmp_H_position[hb_iii][2] > ccell):
                       dcoordH_a.append(m.floor(tmp_H_position[hb_iii][2]/ccell)*ccell)
                    elif (tmp_H_position[hb_iii][2] <= float(0)):
                       dcoordH_a.append(-m.ceil(abs(tmp_H_position[hb_iii][2])/ccell)*ccell)
                    else:
                       dcoordH_a.append(float(0.0))

                    xcoordHnew = tmp_H_position[hb_iii][0]-dcoordH_a[0]
                    ycoordHnew = tmp_H_position[hb_iii][1]-dcoordH_a[1]
                    zcoordHnew = tmp_H_position[hb_iii][2]-dcoordH_a[2]
                    newH_pos = np.array([xcoordHnew, ycoordHnew, zcoordHnew])
                    tmp_H_position[hb_iii] = newH_pos
                    ohbond_dist_new = distances.calc_bonds(coordO, newH_pos, box=None)
                    delta_ohbond_dist = abs(ohbond_dist_new - ohbond_dist_pbc)
                    self.delta_ohbond_dist_list.append(delta_ohbond_dist)
                    if delta_ohbond_dist > 0.5:
                        ohbond_dist_tmp = distances.calc_bonds(np.array([0, coordO[1], 0]), np.array([0, newH_pos[1], 0]), box=None); arbitary_cutoff = 2.0
                        if coordO[1] <= self.box[1]/2 and newH_pos[1] > self.box[1]/2 and ohbond_dist_tmp > arbitary_cutoff:
                            newH_pos[1] = self.box[1] - newH_pos[1] 
                        if coordO[1] >= self.box[1]/2 and newH_pos[1] < self.box[1]/2 and ohbond_dist_tmp > arbitary_cutoff:
                            newH_pos[1] = self.box[1] + newH_pos[1]
                        ohbond_dist_tmp = distances.calc_bonds(np.array([0, 0, coordO[2]]), np.array([0, 0, newH_pos[2]]), box=None)
                        if coordO[2] <= self.box[2]/2 and newH_pos[2] > self.box[2]/2 and ohbond_dist_tmp > arbitary_cutoff:
                            newH_pos[2] = self.box[2] - newH_pos[2]
                        if coordO[2] >= self.box[2]/2 and newH_pos[2] < self.box[2]/2 and ohbond_dist_tmp > arbitary_cutoff:
                            newH_pos[2] = self.box[2] + newH_pos[2]
                    ohbond_dist_new = distances.calc_bonds(coordO, newH_pos, box=None)
                    ohbond_dist_pbc = distances.calc_bonds(coordO, newH_pos, box=self.box) #
                    delta_ohbond_dist = abs(ohbond_dist_new - ohbond_dist_pbc)

        "obtain the OH vectors and dipole vector"
        #OH_Vector_1 = tmp_H_position[0] - d.position   # toward H coordO
        OH_Vector_1 = tmp_H_position[0] - coordO    # toward O
        OH_Vector_2 = tmp_H_position[1] - coordO   # toward H
        #OH_Vector_2 = d.position - tmp_H_position[1]    # toward O
        
        #print(type(tmp_H_position[0]),  tmp_H_position)
        dipVector = (tmp_H_position[0] + tmp_H_position[1]) * 0.5  - coordO 
        #print("[", dipVector[0], ",", dipVector[1], ",", dipVector[2], "], ")
        #self.dipVector_list.append(dipVector)
        
        tmp = 0; tmp_f = 0; tmp_if = 0; tmp_im=0; tmp_bulk=0; tmp_ne=0; tmp_etc=0;
    
        angle_dip_xaxis, cosTheta_dip_xaxis = self.vector2degree(dipVector, 'x')
        angle_dip_yaxis, cosTheta_dip_yaxis = self.vector2degree(dipVector, 'y')
        angle_dip_zaxis, cosTheta_dip_zaxis = self.vector2degree(dipVector, 'z')
        
        angle_OH_1_xaxis, cosTheta_OH_1_xaxis = self.vector2degree(OH_Vector_1, 'x')
        angle_OH_1_yaxis, cosTheta_OH_1_yaxis = self.vector2degree(OH_Vector_1, 'y')
        angle_OH_1_zaxis, cosTheta_OH_1_zaxis = self.vector2degree(OH_Vector_1, 'z')
        
        angle_OH_2_xaxis, cosTheta_OH_2_xaxis = self.vector2degree(OH_Vector_2, 'x')
        angle_OH_2_yaxis, cosTheta_OH_2_yaxis = self.vector2degree(OH_Vector_2, 'y')
        angle_OH_2_zaxis, cosTheta_OH_2_zaxis = self.vector2degree(OH_Vector_2, 'z')

        # variables for feeOH analysis
        S_tmp=0; dd = 0; nd = 0; tmp_freeOH = 0
        SD_parallel=0; SD_perpendicular=0; freeOH_perpendicular=0;freeOH_parallel=0; tmp_d =0
        if self.cutoff_IF[0] < d.position[0] <= self.cutoff_IF[1]: 
            tmp_if += 1

            self.valdip_IF_xaxis.append(angle_dip_xaxis)
            self.valdip_IF_yaxis.append(angle_dip_yaxis)
            self.valdip_IF_zaxis.append(angle_dip_zaxis)
            
            self.cosTheta_valdip_IF_xaxis.append(cosTheta_dip_xaxis)
            self.cosTheta_valdip_IF_yaxis.append(cosTheta_dip_yaxis)
            self.cosTheta_valdip_IF_zaxis.append(cosTheta_dip_zaxis)

            # Angle OH vector
            self.vecOH_IF_xaxis.append(angle_OH_1_xaxis)
            self.vecOH_IF_yaxis.append(angle_OH_1_yaxis)
            self.vecOH_IF_zaxis.append(angle_OH_1_zaxis)
            
            self.vecOH_IF_xaxis.append(angle_OH_2_xaxis)
            self.vecOH_IF_yaxis.append(angle_OH_2_yaxis)
            self.vecOH_IF_zaxis.append(angle_OH_2_zaxis)
           
            # Radian OH vector
            self.cosTheta_vecOH_IF_xaxis.append(cosTheta_OH_1_xaxis)
            self.cosTheta_vecOH_IF_yaxis.append(cosTheta_OH_1_yaxis)
            self.cosTheta_vecOH_IF_zaxis.append(cosTheta_OH_1_zaxis)
            
            self.cosTheta_vecOH_IF_xaxis.append(cosTheta_OH_2_xaxis)
            self.cosTheta_vecOH_IF_yaxis.append(cosTheta_OH_2_yaxis)
            self.cosTheta_vecOH_IF_zaxis.append(cosTheta_OH_2_zaxis)

        elif self.cutoff_BULK[0] < d.position[0] <= self.cutoff_BULK[1]:
            self.cosTheta_valdip_BULK_xaxis.append(cosTheta_dip_xaxis)
            self.cosTheta_vecOH_BULK_xaxis.append(cosTheta_OH_1_xaxis)
            self.cosTheta_vecOH_BULK_xaxis.append(cosTheta_OH_2_xaxis)

        elif d.position[0] > self.cutoff_BULK[1]:
            self.cosTheta_valdip_NE_xaxis.append(cosTheta_dip_xaxis)
            self.cosTheta_vecOH_NE_xaxis.append(cosTheta_OH_1_xaxis)
            self.cosTheta_vecOH_NE_xaxis.append(cosTheta_OH_2_xaxis)
        
        #return (nd, SD_perpendicular, SD_parallel, dd)
        
    def run(self, start, stop, step):

        for ts in self.u.trajectory[start:stop:step]:
            # Select atom groups
            self._s1 = self.u.select_atoms(self.selection1)

            #_update_selection_1() call the function _get_bonded_hydrogens_dist
            #to find the bonded Hydrogens
            self._update_selection_1()

            DD_tmp = 0; SD_perpendicular_tmp = 0; SD_parallel_tmp = 0; ND_tmp = 0;
            for i, donor_h_set in self._s1_donors_h.items():
                
                # i-th donor in the atom roup _s1 (selection1)
                d = self._s1[i]

                for h in donor_h_set:
                    "Find _S2 atoms (or any) around h within self.cutoff_dist_O_H " # **
                    res = self.ns_acceptors.search(h, self.cutoff_dist_O_H)
                    # d-h -- a ; Search a around h within cutoff_dist_O_H 
                   
                    "Net atomic charge: obtain the total NAC for a water"
                    #print("H: n_frame: %s, pos: %s, charge: %s" %( self.n_frames-1,self.nac_water_pos_charge[self.n_frames-1][0][h.index] ,  self.nac_water_pos_charge[self.n_frames-1][1][h.index]  )) 
                    tmp_net_charge_water += self.nac_water_pos_charge[self.n_frames-1][1][h.index] 

                    for a in res:
                            #print(d.index, a.index) 
                            if d.index == a.index:
                                #print("duplicated pairs donor %s acceptor %s" %(d.index, a.index))
                                #print("skip...\n")
                                #continue
                                pass
                            else:
                                #print("Calc angle between donor %s and acceptor %s" %(d.index, a.index))
                                #print("Main loop: indexes of the d-h-a: %s-%s-%s " %(d.index, h.index, a.index))
                                
#                                if (d.index, h.index, a.index) in already_found or (a.index, h.index, d.index) in already_found:
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
                                    
                                    dist_1 = distances.calc_bonds(h.position, d.position, box=self.box)
                                    dist_2 = distances.calc_bonds(h.position, a.position, box=self.box)

                                    dist_delta = abs(dist_1 - dist_2)
                                    
                                    #if dist_delta < 0.1:
                                        #print("frame: ", ts.frame)
                                        #print("dist delta: ", dist_delta, dist_1, dist_2)
                                        #print("H-O(d)-O(a) %s-%s-%s" %(h.index, d.index, a.index))
                                    #    print("O(d) position: ", d.position)
                                    #    print("O(a) position: ", a.position)
                                    #    print("[%s(h)-%s(d)-%s(a)] dist: %s, angle: %s" %(h.index, d.index, a.index, dist, angle))

                                    dist_O1_H = distances.calc_bonds(d.position, h.position, box=self.box)
                                    dist_O2_H = distances.calc_bonds(a.position, h.position, box=self.box)

                                   
                                    "Rectangular HBs cutoffs"
                                    if self.HBs_criteria == 'Luzar':
                                        #print("Luzar")
                                        if angle <= self.angle and dist <= self.cutoff_dist_donor_acceptor:
                                            #print("****************** hydrogen position: %s\n  Donor position: %s\n, acceptor position: %s" %(h.position, d.position, a.position))
                                        
                                            #print("****************** HB Found: %s-%s-%s; dist:%s angle: %s" %(d.index, h.index, a.index, dist, angle))
                                            #print("****************** location: ", h.position[0],d.position[0], a.position[0])
                                            #print("the distance %s %s" %(dist_O1_H, dist_O2_H))
                                            #print("the angle %s" %(angle))
                                            frame_results.append( [d.index, h.index, a.index, (d.name, h.name, a.name), h.position[0], dist, angle])
                                            already_found[(d.index, h.index, a.index)] = True
 
                                            #if d.index == 23: 
                                            #    print("***** d-h-a: ", d.index, h.index, a.index, "*****")
                                            #    print("the distance %s %s" %(dist_O1_H, dist_O2_H))
                                            #    print("the angle %s" %(angle))
                                            #    input('enter...')
                                            #if d.index == 221 and h.index == 219: #and a.index == 179: 
                                            #   print("***** d-h-a: ", d.index, h.index, a.index, "*****")                                      
                                            # Make a new dict for analysis
                                            # Add the dornor and acceptor pair into the dict.
                                            # Add new value to the list of the key if a key does not exist, make a new key
                                            dict_d_a.setdefault(d,[]).append(a)
                                            dict_a_d.setdefault(a.index,[]).append(d)
                                            c += 1
                                            
                                            "#####***** HBs ACF (IF) *****#####"
                                            if self.cutoff_IF[0] < d.position[0] <= self.cutoff_IF[1]:
                                                if frame == self.start:
                                                    prev_already_found_IF[(d.index, h.index, a.index)] = True
                                                else:
                                                    #print("----- NOT first frame-----")
                                                    # search the same d-h-a, which survive from the previous 
                                                    if (d.index, h.index, a.index) in prev_already_found_IF:
                                                        hb_acf_already_found_IF[(d.index, h.index, a.index)] = True
                                                        self.hb_acf_results_IF[self.n_frames-1] += 1

                                            "#####***** HBs ACF (BULK) *****#####"
                                            if d.position[0] >= self.cutoff_BULK[0] and d.position[0] <= self.cutoff_BULK[1]:
                                                if frame == self.start:
                                                    prev_already_found_BULK[(d.index, h.index, a.index)] = True
                                                else:
                                                    if (d.index, h.index, a.index) in prev_already_found_BULK:
                                                        hb_acf_already_found_BULK[(d.index, h.index, a.index)] = True
                                                        self.hb_acf_results_BULK[self.n_frames-1] += 1
                                                
                                            #"#####***** HBs ACF *****#####"
                                            #if frame != self.start:
                                            #    #print("----- NOT first frame-----")
                                            #    # search the same d-h-a, which survive from the previous 
                                            #    if (d.index, h.index, a.index) in prev_already_found:
                                            #        hb_acf_already_found[(d.index, h.index, a.index)] = True
                                            #        self.hb_acf_results[self.n_frames-1] += 1
                                            
                                                
                                                
                                            
                                    #"Triangular HBs cutoffs"    
                                    elif self.HBs_criteria == 'Sho':
                                        #print("Sho")
                                        cosine_term = -1.71 * np.cos(angle_OHO_rad) + 1.37
                                        #print("dist O_H: ", dist_O2_H, "cosine_term: ", cosine_term)
                                        if dist_O2_H < cosine_term:
                                            #print("****************** HB (Sho) Found: dist h-o(a) %s cos %s" %(dist_O2_H, cosine_term))
                                            #print("****************** location: ", h.position[0],d.position[0], a.position[0])
                                            #print("the distance %s %s" %(dist_O1_H, dist_O2_H))
                                            #print("the angle %s" %(angle))
                                            frame_results.append( [d.index, h.index, a.index, (d.name, h.name, a.name), h.position[0], dist, angle])
                                            already_found[(d.index, h.index, a.index)] = True
                                        
                                            # Make a new dict for analysis
                                            # Add the dornor and acceptor pair into the dict.
                                            # Add new value to the list of the key if a key does not exist, make a new key
                                            dict_d_a.setdefault(d,[]).append(a)
                                            dict_a_d.setdefault(a.index,[]).append(d)
                                            c += 1
                # End of h loop
                t2=time.time()
                dt=t2-t1
                self.t_l.append(dt)
                "Net atomic charge"
                self.water_tot_nac_pos.append(self.nac_water_pos_charge[self.n_frames-1][0][d.index])
                self.water_tot_nac_charge.append(tmp_net_charge_water)
                #print(self.water_tot_nac_charge)
                #input('enter...')

            # End of d loop
            "###********** HB ACF (IF) ***********###"
            if frame == self.start:
                already_found_first_frame_IF = prev_already_found_IF
                len_hb_acf_firstFrame_IF = len(already_found_first_frame_IF)
                self.hb_acf_results_IF[self.n_frames-1] = len_hb_acf_firstFrame_IF
                #prev_already_found_IF = already_found_IF
            else:
                prev_already_found_IF = hb_acf_already_found_IF
                
            "###********** HB ACF (BULK) ***********###"
            if frame == self.start:
                already_found_first_frame_BULK = prev_already_found_BULK
                len_hb_acf_firstFrame_BULK = len(already_found_first_frame_BULK)
                self.hb_acf_results_BULK[self.n_frames-1] = len_hb_acf_firstFrame_BULK
                #prev_already_found_IF = already_found_IF
            else:
                prev_already_found_BULK = hb_acf_already_found_BULK
                
            #"###********** HB ACF ***********###"
            #if frame == self.start:
            #    already_found_first_frame = already_found
            #    len_hb_acf_firstFrame = len(already_found_first_frame)
            #    self.hb_acf_results[self.n_frames-1] = len_hb_acf_firstFrame 
            #    prev_already_found = already_found
            #else:
            #    prev_already_found = hb_acf_already_found
            
            #print(already_found_first_frame_IF)
            
            #else:
            #    self.hb_acf_results[self.n_frames] = self.hb_acf_results[self.n_frames] / len_hb_acf_firstFrame
            #print("----------------")   # (ff)
            #print("len HB_ACF: {}".format(len(hb_acf_already_found)))
            #print("HB ACF results: %s" %(self.hb_acf_results))
            #print("----------------")
            
            t2=time.time()
            dt=t2-t1
            self.t_l.append(dt)



            "---------- HBs analysis ----------"
            t1=time.time()
            # [within frame] Update the count
            # Water at IF gives 2 or 1 HBs. 
            count_IF_s=0; count_IF_a=0; count_IM_s=0; count_IM_a=0; count_BULK_s=0; count_BULK_a=0;
            count_NE_s=0; count_NE_a=0
            
            # Loop over all the HB at this frame
            # Key is the donor and value is the acceptors.
            # If donor have two acceptors, value would be [atom_group1, atom_group2]
            # If donor have one acceptor, value would be [atom_group1]
            # Or, if donor have no acceptor, value would be [].
            # Then, the position of atom_group1/atom_group2 can be accessed.
            
            #print("Total Nr. of water molecules that have HBs", len(dict_d_a))
            
            "Total Nr. of water molecules that have no HBs"
            "The oxygens which are not in dict_d_a is the oxygens do not donate H"
            water_mols_non_HBs =  s1_tot_res - len(dict_d_a) 
            
            " Statistics: use 2x2 array to count HBs wrt positinos of the donors"
            tmp = 0; tmp_f = 0; tmp_if = 0; tmp_im=0; tmp_bulk=0; tmp_ne=0; tmp_etc=0;
            #print("len(dict_d_a): %s, dict_a_d: %s, frame_results: %s " % (len(dict_d_a), len(dict_a_d), len(frame_results) ))
            for key, value in dict_d_a.items():
                #key: donor, values: accepting water molecules
                tmp += 1
                # Get the position of the donor
                donor_pos = key.position[0]
                donor_pos_xyz = key.position

                donor_index = key.index
                donor_atomGroup = self._s1[self.index_to_i(donor_index)]
                #print("i %s - index %s " %(i, self.i_to_index(i)  ))

                if key.index in dict_a_d:
                    tmp_donor_as_acceptor = dict_a_d[key.index]
                else:
                    tmp_donor_as_acceptor = []
                #print("  --- D %s, A %s ---" %(len(value),len(tmp_donor_as_acceptor) ))
                #if donor_index == 2:
                #    print("donor index: %s, position %s chg: %s" %(donor_index, donor_pos, self.nac_water_pos_charge[self.n_frames-1][1][donor_index])); input('enter')
                #print("index %s - position %s " %(donor_index, self._s1[self.index_to_i(donor_index)].position)) 
                #if donor_index == 509:
                #    print(" donor_index: ", donor_index)
                #    print("index %s - position %s " %(donor_index, key.position)) 
                #    print("D%s ", len(value))
                #    input('enter...')

                #for val in range(len(value)):
                #    print("donor(%s)-acceptors: %s" %(donor_index, value[val].index))
                net_water_chg = 0
                #input('enter...')
                " Donors at IF "
                if  self.cutoff_IF[0] < donor_pos <= self.cutoff_IF[1]:
                    #print("IF")
                    tmp_if +=1
                    if len(value) >= 2:
                        #print("Donor at IF, donate 2")
                        #print("donor index: %s, position %s" %(donor_index, donor_pos))

                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            #print("Accept hydrogens from: ",  donor_as_acceptor)    # give a list of oxygens that donate H to the reference O.

                            #input("enter...")
                            #print("Number of HBs accepted: %s " %( len(donor_as_acceptor)))
                            if len(donor_as_acceptor) >= 2:            # D2A2
                                #print("D2A2")
                                " HB statistics EXT. "
                                count_Dss, count_Ass, Asn0 = _update_DAss(value, donor_as_acceptor)    # update the number of ss/nn/ns 
                                _update_count_array_IF_EXT_D2A2(count_Dss, count_Ass)              # update the extended HB matrix

                                " HB statistics "
                                ndx_HB_n = 0; ndx_HB_m = 2
                                count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                # Call oxygen position and angle  
                                tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]     # [0]: position (x) 
                                tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                                self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                #self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append((donor_pos_netWaterchg[0], donor_pos_netWaterchg[1])) 
                                self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg) 
                                # Add donor index for vmd
                                fout_HBndx_d2ax.write("%s " %donor_index)
                                #print("index: %s" %key.index)

                                #print("D2A2 ref. index: %s, Donors %s)" %(donor_index, donor_as_acceptor))                               
                                " Collect HB D2A2 "
                                self.HB_collect_d2ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    "Collect HB D2Ax only H"
                                    #print("D2A2 - Oindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                    CheckH = HB_collect_check_H(donor_pos_xyz, donor_h_set_nac.position)
                                    #print('D2ax checkH: ', CheckH)
                                    #input('enter')
                                    if CheckH[0] == True:
                                        self.HB_collect_d2ax_HposAngle.append(donor_h_set_nac.index)
                                    if CheckH[1] == True:
                                        self.HB_collect_d2ax_HAngle.append(donor_h_set_nac.index)
                                    if CheckH[2] == True:
                                        self.HB_collect_d2ax_HAngle40.append(donor_h_set_nac.index)

                                    count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    #print("--- h.index: %s, h.position: %s ---" %( donor_h_set_nac.index, donor_h_set_nac.position)) 
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    #print("----------")
                                    #print("h.index %s, %s" %(donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index]))
                                    
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                        #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                        tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                        tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                        tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                        _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg) 
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call all h.index"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_xaxis_HB_tuple_IF: 
                                        tmp_h_angle = self.hydrogen_pos_angle_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]  # [0]: position (x) 
                                        _update_charge_H_pos_angle_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_angle)
                                    #if key.index == 221: #and a.index == 179:
                                    #    for val in range(len(value)):
                                    #        print("D2A2-donor(%s)-acceptors: %s" %(donor_index, value[val].index))
                                    #        input('enter...')

 
                            elif len(donor_as_acceptor) == 1:
                                " HB statistics EXT. "
                                count_Dss, count_Ass, Asn0 = _update_DAss(value, donor_as_acceptor)    # update the number of ss/nn/ns 
                                _update_count_array_IF_EXT_D2A1(count_Dss, count_Ass, Asn0)              # update the extended HB matrix

                                ndx_HB_n = 0; ndx_HB_m = 1
                                count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                # Call oxygen position and angle  
                                tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                                tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                                self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)
                                # Add donor index for vmd
                                fout_HBndx_d2ax.write("%s " %donor_index)

                                #print("D2A1 ref. index: %s, Donors %s)" %(donor_index, donor_as_acceptor))
                                " Collect HB D2A1 "
                                self.HB_collect_d2ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    "Collect HB D2A1 only H"
                                    #print("D2A1 - Oindex: %s, H index: %s, Opos: %s, Hpos: %s" %(donor_index,donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                    CheckH = HB_collect_check_H(donor_pos_xyz, donor_h_set_nac.position)
                                    #input('donor_h_set_nac.index %s, enter...'%(donor_h_set_nac.index))
                                    if CheckH[0] == True:
                                        self.HB_collect_d2ax_HposAngle.append(donor_h_set_nac.index)
                                    if CheckH[1] == True:
                                        self.HB_collect_d2ax_HAngle.append(donor_h_set_nac.index)
                                    if CheckH[2] == True:
                                        self.HB_collect_d2ax_HAngle40.append(donor_h_set_nac.index)

                                    count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                        #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                        tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                        tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                        tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                        _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call all h.index"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_xaxis_HB_tuple_IF: 
                                        tmp_h_angle = self.hydrogen_pos_angle_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]  # [0]: position (x) 
                                        _update_charge_H_pos_angle_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_angle)
                                    #if key.index == 221: # and a.index == 179:
                                    #    for val in range(len(value)):
                                    #        print("D2A1-donor(%s)-acceptors(%s)-hydrogen(%s)" %(donor_index, value[val].index, donor_h_set_nac.index))
                                    #        input('enter...')


                        else:
                            " HB statistics EXT. "
                            count_Dss, Asn0 = _update_D2ss(value)    # update the number of ss/nn/ns 
                            _update_count_array_IF_EXT_D2A0(count_Dss, Asn0)              # update the extended HB matrix

                            #print("D2 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 0; ndx_HB_m = 0
                            count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                            count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                            # Call oxygen position and angle  
                            tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                            tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                            self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                            " NAC of water and position(x) "
                            donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                            self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)
                            # Add donor index for vmd
                            fout_HBndx_d2ax.write("%s " %donor_index)

                            #print("D2A0 ref. index: %s, Donors %s)" %(donor_index, donor_as_acceptor))
                            " Collect HB D2Ax "
                            self.HB_collect_d2ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out
                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                "Collect HB D2A0 only H"
                                #print("D2A0 - Oindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                CheckH = HB_collect_check_H(donor_pos_xyz, donor_h_set_nac.position)
                                if CheckH[0] == True:
                                    self.HB_collect_d2ax_HposAngle.append(donor_h_set_nac.index)
                                if CheckH[1] == True:
                                    self.HB_collect_d2ax_HAngle.append(donor_h_set_nac.index)
                                if CheckH[2] == True:
                                    self.HB_collect_d2ax_HAngle40.append(donor_h_set_nac.index)

                                count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                    #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                    tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                    tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                    tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                    _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)

                                #if key.index == 221: #and a.index == 179:
                                #    for val in range(len(value)):
                                #        print("D2A0-donor(%s)-acceptors: %s" %(donor_index, value[val].index))
                                #        input('enter...')
                              
                    if len(value) == 1:
                        #print("Donor at IF, donate 1")
                        
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            #print("donor index: %s" %(donor_index))
                            #print("donor index: %s, acceptor: %s" %(donor_index, donor_as_acceptor))
                            #print("Number of HBs accepted: %s " %( len(donor_as_acceptor)))
                            if len(donor_as_acceptor) >= 2:
                                #print("\nD1A2")
                                " HB statistics EXT. "
                                count_Dss, count_Ass, Asn0 = _update_DAss(value, donor_as_acceptor)    # update the number of ss/nn/ns 
                                _update_count_array_IF_EXT_D1A2(count_Dss, count_Ass)              # update the extended HB matrix
                                #input('enter...')

                                "HB matrix"
                                ndx_HB_n = 1; ndx_HB_m = 2
                                count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                # Call oxygen position and angle  
                                tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                                tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                                self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)
                                # Add donor index for vmd
                                fout_HBndx_d1ax.write("%s " %donor_index)
                                #print(key.index, key.position)   # key.position the reference atom;  #value[val].index: the accepting o index
                                #HB_d1ax_collect_coord_i(self.index_to_i(donor_index))   # hold on. 

                                " Collect HB D1A2 "
                                #self.HB_collect_d1ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out 
                                hb_d1ax_dangling_h = False
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    "Collect HB D1A2 only H"
                                    #print("*** D1A2 Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                    if (donor_index, donor_h_set_nac.index, value[0].index) in already_found:
                                        pass
                                    else: 
                                        hb_d1ax_dangling_h = True
                                        #for donors_for_ref in donor_as_acceptor:
                                        #    print(donors_for_ref.index)

                                    if hb_d1ax_dangling_h == True:
                                        # collect all d1ax
                                        self.HB_collect_d1ax_dangling.append(donor_h_set_nac.index)
                                        self.HB_collect_d1a2_dangling.append(donor_h_set_nac.index)
                                        #print('dangling H: ', donor_h_set_nac.index); input('enter')

                                    CheckH = HB_collect_check_H(donor_pos_xyz, donor_h_set_nac.position)
                                    if CheckH[0] == True and hb_d1ax_dangling_h == True:
                                        #self.HB_collect_d1ax_HposAngle.append(donor_h_set_nac.index)
                                        IF_nac_H_d1a2_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    if CheckH[1] == True and hb_d1ax_dangling_h == True:
                                    #    self.HB_collect_d1ax_HAngle.append(donor_h_set_nac.index)
                                    #    self.HB_collect_d1ax_nonbondedH.append(donor_h_set_nac.index)
                                        #print("=== D1A2 Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                        IF_nac_H_d1a2_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    #if CheckH[2] == True:
                                    #    self.HB_collect_d1ax_HAngle40.append(donor_h_set_nac.index)
                                    if CheckH[3] == True and hb_d1ax_dangling_h == True:
                                    #    self.HB_collect_d1ax_HposAngle_window.append(donor_h_set_nac.index)
                                    #    self.HB_collect_d1ax_nonbondedH.append(donor_h_set_nac.index)
                                        #print("=== D1A2 Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                        IF_nac_H_d1a2_nonbonded_tilt.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    #input('enter..')
                                    hb_d1ax_dangling_h = False

                                    count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                        #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                        tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                        tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                        tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                        tmp_au_near_h_index = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][3]

                                        #print("H index: %s, Au index: %s, Au charge: %s" %(donor_h_set_nac.index, tmp_au_near_h_index, tmp_au_near_h_averchg))
                                        #input('enter..')
                                        _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)

                                    "Call the H-O angle with h.index w.r.t the HB matrix; call all h.index"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_xaxis_HB_tuple_IF: 
                                        tmp_h_angle = self.hydrogen_pos_angle_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]  # [0]: position (x) 
                                        _update_charge_H_pos_angle_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_angle)


                                #input('enter')
                            elif len(donor_as_acceptor) == 1:
                                " HB statistics EXT. "
                                #print("\nD1A1")
                                count_Dss, count_Ass, Asn0 = _update_DAss(value, donor_as_acceptor)    # update the number of ss/nn/ns 
                                _update_count_array_IF_EXT_D1A1(count_Dss, count_Ass, Asn0)              # update the extended HB matrix
                                
                                ndx_HB_n = 1; ndx_HB_m = 1 
                                count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                # Call oxygen position and angle  
                                tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                                tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                                self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)
                                # Add donor index for vmd
                                fout_HBndx_d1ax.write("%s " %donor_index)

                                " Collect HB D1A1 "
                                #self.HB_collect_d1ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out
                                hb_d1ax_dangling_h = False
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    "Collect HB D1A1 only H"
                                    #print("*** D1A1 Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                    if (donor_index, donor_h_set_nac.index, value[0].index) in already_found:
                                        pass
                                    else: 
                                        hb_d1ax_dangling_h = True
                                        #print("-- D1A1 Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0])); input('enter')

                                    if hb_d1ax_dangling_h == True:
                                        # collect all d1ax
                                        self.HB_collect_d1ax_dangling.append(donor_h_set_nac.index)
                                        #print('h dangling index: ', donor_h_set_nac.index); input('enter')
                                        self.HB_collect_d1a1_dangling.append(donor_h_set_nac.index)

                                    CheckH = HB_collect_check_H(donor_pos_xyz, donor_h_set_nac.position)
                                    if CheckH[0] == True and hb_d1ax_dangling_h == True:
                                        self.HB_collect_d1ax_HposAngle.append(donor_h_set_nac.index)
                                        IF_nac_H_d1a1_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("dict h.index: %s,  h.pos: %s, charge: %s" %( donor_h_set_nac.index, donor_h_set_nac.position,  self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )); input('enter') 
                                        #print("=== D1A1 HposAngle Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))

                                    if CheckH[1] == True and hb_d1ax_dangling_h == True:
                                        self.HB_collect_d1ax_HAngle.append(donor_h_set_nac.index)
                                        self.HB_collect_d1ax_nonbondedH.append(donor_h_set_nac.index)
                                        IF_nac_H_d1a1_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== D1A1 HAngle Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                    if CheckH[2] == True:
                                        self.HB_collect_d1ax_HAngle40.append(donor_h_set_nac.index)
                                        #print("=== D1A1 HAngle40 Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                    if CheckH[3] == True and hb_d1ax_dangling_h == True:
                                        self.HB_collect_d1ax_HposAngle_window.append(donor_h_set_nac.index)
                                        self.HB_collect_d1ax_nonbondedH.append(donor_h_set_nac.index)
                                        IF_nac_H_d1a1_nonbonded_tilt.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== D1A1 Window Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))


                                    hb_d1ax_dangling_h = False
                                    count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                        #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                        tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                        tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                        tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                        _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call all h.index"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_xaxis_HB_tuple_IF: 
                                        tmp_h_angle = self.hydrogen_pos_angle_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]  # [0]: position (x) 
                                        _update_charge_H_pos_angle_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_angle)  
                                 
                        else:
                            " HB statistics EXT. "
                            count_Dss, Asn0 = _update_D2ss(value)    # update the number of ss/nn/ns 
                            _update_count_array_IF_EXT_D1A0(count_Dss, Asn0)              # update the extended HB matrix

                            #print("D1 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 1; ndx_HB_m = 0
                            count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                            count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                            # Call oxygen position and angle  
                            tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                            tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                            self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                            " Collect HB D1A0 "
                            #self.HB_collect_d1ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out
                            #" NAC of water and position(x) "
                            donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                            self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)
                            # Add donor index for vmd
                            fout_HBndx_d1ax.write("%s " %donor_index)

                            hb_d1ax_dangling_h = False
                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                "Collect HB D1A0 only H"
                                #print("*** D1A0 Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                if (donor_index, donor_h_set_nac.index, value[0].index) in already_found:
                                    pass
                                else: 
                                    hb_d1ax_dangling_h = True
                                    #print("-- D1A0 Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))

                                if hb_d1ax_dangling_h == True:
                                    # collect all d1ax
                                    self.HB_collect_d1ax_dangling.append(donor_h_set_nac.index) 
                                    self.HB_collect_d1a0_dangling.append(donor_h_set_nac.index)

                                #CheckH = HB_collect_check_H(donor_pos_xyz, donor_h_set_nac.position)
                                #if CheckH[0] == True and hb_d1ax_dangling_h == True:
                                #    self.HB_collect_d1ax_HposAngle.append(donor_h_set_nac.index)
                                #if CheckH[1] == True and hb_d1ax_dangling_h == True:
                                #    self.HB_collect_d1ax_HAngle.append(donor_h_set_nac.index)
                                #    self.HB_collect_d1ax_nonbondedH.append(donor_h_set_nac.index)
                                #    #print("=== D1A0 Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                #if CheckH[2] == True:
                                #    self.HB_collect_d1ax_HAngle40.append(donor_h_set_nac.index)
                                #if CheckH[3] == True and hb_d1ax_dangling_h == True:
                                #    self.HB_collect_d1ax_HposAngle_window.append(donor_h_set_nac.index)
                                #    self.HB_collect_d1ax_nonbondedH.append(donor_h_set_nac.index)
                                #    #print("=== D1A0 Oindex: %s, nonboned Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))
                                hb_d1ax_dangling_h = False

                                count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                    #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                    tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                    tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                    tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                    _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)                           
                # Donors at IM % !!! not updated for the >= 3
                #elif donor_pos > self.cutoff_IF and donor_pos < self.cutoff_BULK[0]:
                elif  self.cutoff_IF[1] < donor_pos < self.cutoff_BULK[0]:
                    #print("IM")
                    tmp_im += 1
                    if len(value) >= 2:
                        #print("Donor at IF, donate 2")
                        
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            #print("Number of HBs accepted: %s " %( len(donor_as_acceptor)))
                            if len(donor_as_acceptor) >= 2:
                                ndx_HB_n = 0; ndx_HB_m = 2
                                count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
               
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    #"NAC for hydrogen charge when H is not bridged"
                                    #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            elif len(donor_as_acceptor) == 1:
                                ndx_HB_n = 0; ndx_HB_m = 1
                                count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    #"NAC for hydrogen charge when H is not bridged"
                                    #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                        else:
                            #print("D1 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 0; ndx_HB_m = 0
                            count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                            count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                #"NAC for hydrogen charge when H is not bridged"
                                #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                               
                    if len(value) == 1:
                        #print("Donor at IF, donate 1")
                        
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            #print("Number of HBs accepted: %s " %( len(donor_as_acceptor)))
                            if len(donor_as_acceptor) >= 2:
                                ndx_HB_n = 1; ndx_HB_m = 2
                                count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    #"NAC for hydrogen charge when H is not bridged"
                                    #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)


                            elif len(donor_as_acceptor) == 1:
                                ndx_HB_n = 1; ndx_HB_m = 1
                                count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    #"NAC for hydrogen charge when H is not bridged"
                                    #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                        else:
                            #print("D1 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 1; ndx_HB_m = 0
                            count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                            count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                #"NAC for hydrogen charge when H is not bridged"
                                #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                           
                # Donors at BULK
                elif donor_pos >= self.cutoff_BULK[0] and donor_pos <= self.cutoff_BULK[1]:
                    tmp_bulk += 1; 

                    if len(value) >= 3:
                        #print("Donor at BULK, D %s" %len(value))
                        #print("Index: %s , Position: %s " %( key.index, donor_pos))
                        #print("index from atom group: %s, position: %s " %( donor_atomGroup.index, donor_atomGroup.position[0]))
                        #input("Enter..")
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            if len(donor_as_acceptor) >= 3:
                                ndx_HB_n = 0; ndx_HB_m = 3
                                #print("D3 A3 acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    
                            elif len(donor_as_acceptor) == 2:
                                #print("D3-A2: acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                #input('enter...')
                                ndx_HB_n = 0; ndx_HB_m = 2
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                #print("donor index: ", donor_index)
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            elif len(donor_as_acceptor) == 1:
                                #print("D3 A1 acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                ndx_HB_n = 0; ndx_HB_m = 1
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                        else:
                            #print("D1 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 0; ndx_HB_m = 0
                            count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                            count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])

                            " NAC of water and position(x) "
                            donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                            self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                    if len(value) == 2:
                        
                        #print("Donor at BULK, D %s" %len(value))
                        #print("Index: %s , Position: %s " %( key.index, donor_pos))
                        
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            if len(donor_as_acceptor) >= 3:
                                ndx_HB_n = 1; ndx_HB_m = 3
                                #print(" D2-A3  acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            elif len(donor_as_acceptor) == 2:
                                #print(" D2-A2  acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                ndx_HB_n = 1; ndx_HB_m = 2
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            elif len(donor_as_acceptor) == 1:
                                #print(" D2-A1  acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                #print("D2-A1: acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                #input('enter...')
                                ndx_HB_n = 1; ndx_HB_m = 1
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                #print("")
                                #print("  donor O Index: %s, i: %s, Position: %s " %( key.index, self.index_to_i(donor_index), donor_pos))
                                
                                #if re.match(self.search_val, str(self.nac_water_pos_charge[self.n_frames-1][0][donor_index])) :
                                ##    print("  donor O index from atom group: %s, position: %s " %( donor_atomGroup.index, donor_atomGroup.position[0]))
                                ##    print("  position  a-d: %s - %s " %(donor_atomGroup.position[0], donor_as_acceptor[0].position))
                                #    print("   nac position: ", self.nac_water_pos_charge[self.n_frames-1][0][donor_index])
                                #    #print("   nac charge:   ", self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                #    input("  enter.... ")
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1 
                                #print(count_array_BULK_charge_O)
                                #print(count_array_BULK_charge_O_count)
                                #print("        hydrogens in the acceptor: %s"%(self._s1_donors_h[self.index_to_i(donor_index)]))
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])


                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                        else:
                            #print("D1 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 1; ndx_HB_m = 0
                            count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                            count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])

                            " NAC of water and position(x) "
                            donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                            self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                    if len(value) == 1:
                        #print("Donor at BULK, D %s" %len(value))
                        #print("D1 -- number of HBs accepted: %s " %( len(donor_as_acceptor)))
                       
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            #print("Number of HBs accepted: %s " %( len(donor_as_acceptor)))
                            if len(donor_as_acceptor) >= 3:
                                #print("   acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                ndx_HB_n = 2; ndx_HB_m = 3
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            elif len(donor_as_acceptor) == 2:
                                #print("   acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                #print("D1-A2: acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                #input('enter...')
                                
                                ndx_HB_n = 2; ndx_HB_m = 2
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)


                            elif len(donor_as_acceptor) == 1:
                                #print("   acceptor (%s)  donor (%s)" %(donor_index, self._get_index(donor_as_acceptor)))
                                ndx_HB_n = 2; ndx_HB_m = 1
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                        else:
                            #print("D1 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 2; ndx_HB_m = 0
                            count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                            count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])

                            " NAC of water and position(x) "
                            donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                            self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                            
                # Donors near Ne 
                elif donor_pos > self.cutoff_BULK[1]:
                    #print("NE") # confirmed
                    #print("donor index: %s, pos: %s, %s (ddec),  NAC: %s" %(donor_index, donor_pos_xyz, self.nac_water_pos_charge[self.n_frames-1][0][donor_index],  self.nac_water_pos_charge[self.n_frames-1][1][donor_index]))
                    #print("donor index: %s, pos: %s, %s (ddec),  NAC: %s" %(donor_index, donor_pos_xyz, self.nac_water_pos_charge[self.n_frames-1][0][donor_index],  self.nac_water_pos_charge[self.n_frames-1][1][donor_index]))

                    #input("enter...")

                    tmp_ne += 1
                    if len(value) >= 2:
                        #print("Donor at Ne, donate 2")
                        
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            #print("Number of HBs accepted: %s " %( len(donor_as_acceptor)))
                            if len(donor_as_acceptor) >= 2:
                                " HB statistics "
                                ndx_HB_n = 0; ndx_HB_m = 2
                                count_array_NE[ndx_HB_n][ndx_HB_m] += 1
                                count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += 1
                                #print("O index: %s,  water pos: %s, charge: %s" %( donor_index, self.nac_water_pos_charge[self.n_frames-1][0][donor_index],  self.nac_water_pos_charge[self.n_frames-1][1][donor_index]))
                                self.count_array_NE_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                ###
#                               " Collect HB (Ne) D2A2 "
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    #print("*** (Ne) D2Ax Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0])); input('enter')
                                    #print("dict h.index: %s,  h.pos: %s, charge: %s" %( donor_h_set_nac.index, donor_h_set_nac.position,  self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )); input('enter')
                                    self.HB_collect_d2ax_allH_Ne.append(donor_h_set_nac.index)
                                    #count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    #count_array_NE_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_NE(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    #"Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                    #if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                    #    #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                    #    tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                    #    tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                    #    tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                    #    _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)
                                    #"Call the H-O angle with h.index w.r.t the HB matrix; call all h.index"
                                    #if donor_h_set_nac.index in self.hydrogen_pos_angle_xaxis_HB_tuple_IF: 
                                    #    tmp_h_angle = self.hydrogen_pos_angle_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]  # [0]: position (x) 
                                    #    _update_charge_H_pos_angle_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_angle)


                            elif len(donor_as_acceptor) == 1:
                                ndx_HB_n = 0; ndx_HB_m = 1
                                count_array_NE[ndx_HB_n][ndx_HB_m] += 1
                                count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_NE_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                #self.HB_collect_d1ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out
#                               " Collect HB (Ne) D2A1 "
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    #print("*** (Ne) D2Ax Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0])); input('enter')
                                    self.HB_collect_d2ax_allH_Ne.append(donor_h_set_nac.index)
                                    #count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    #count_array_NE_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_NE(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                        else:
                            ndx_HB_n = 0; ndx_HB_m = 0
                            count_array_NE[ndx_HB_n][ndx_HB_m] += 1
                            count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_NE_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index]) 
                            #" Collect HB D1Ax "
                            ##self.HB_collect_d1ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out
#                           " Collect HB (Ne) D2A0 "
                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                #print("*** (Ne) D2Ax Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0])); input('enter')
                                self.HB_collect_d2ax_allH_Ne.append(donor_h_set_nac.index)
                                self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_NE(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)



                            #for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                            #    #count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                            #    #count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                            #    self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])

                    if len(value) == 1:
                        #print("Donor at IF, donate 1")
                        
                        if key.index in dict_a_d:
                            donor_as_acceptor = dict_a_d[key.index]
                            #print("Number of HBs accepted: %s " %( len(donor_as_acceptor)))
                            if len(donor_as_acceptor) >= 2:
                                ndx_HB_n = 1; ndx_HB_m = 2
                                count_array_NE[ndx_HB_n][ndx_HB_m] += 1
                                count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_NE_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index]) 
                                "Collect H D1A2"
                                hb_d1ax_dangling_h = False
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    #print("*** (Ne) D1A2 Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0]))#; input('enter')
                                    self.HB_collect_d1ax_allH_Ne.append(donor_h_set_nac.index)
                                    self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_NE(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                                    if (donor_index, donor_h_set_nac.index, value[0].index) in already_found:
                                        pass
                                    else: 
                                        hb_d1ax_dangling_h = True

                                    if hb_d1ax_dangling_h == True:
                                        self.HB_collect_d1a2_dangling_Ne.append(donor_h_set_nac.index)
                                        #print('dangling H: ', donor_h_set_nac.index); input('enter')

                                    CheckH = HB_collect_check_H_NE(donor_pos_xyz, donor_h_set_nac.position)
                                    if CheckH[0] == True and hb_d1ax_dangling_h == True:
                                        NE_nac_H_d1a2_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== NE D1A2 pdb Window Oindex: %s, nonboned Hindex: %s, Hpos: %s, H chg: %s" %(donor_index, donor_h_set_nac.index, donor_h_set_nac.position[0], self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )) ; input('enter')
                                    if CheckH[1] == True and hb_d1ax_dangling_h == True:
                                        NE_nac_H_d1a2_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== NE D1A2 pdb Window Oindex: %s, nonboned Hindex: %s, Hpos: %s, H chg: %s" %(donor_index, donor_h_set_nac.index, donor_h_set_nac.position[0], self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )) ; input('enter')
                                    if CheckH[3] == True and hb_d1ax_dangling_h == True:
                                        NE_nac_H_d1a2_nonbonded_tilt.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== NE D1A2 tilt Window Oindex: %s, nonboned Hindex: %s, Hpos: %s, H chg: %s" %(donor_index, donor_h_set_nac.index, donor_h_set_nac.position[0], self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )) ; input('enter')


                                    hb_d1ax_dangling_h = False




                            #    for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                            #        #count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                            #        #count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                            #        self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])

                            elif len(donor_as_acceptor) == 1:
                                ndx_HB_n = 1; ndx_HB_m = 1
                                count_array_NE[ndx_HB_n][ndx_HB_m] += 1
                                count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_NE_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])

                                "Collect H D1A1"
                                hb_d1ax_dangling_h = False
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    #print("*** (Ne) D1A1 Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0])); input('enter')
                                
                                    self.HB_collect_d1ax_allH_Ne.append(donor_h_set_nac.index)
                                    self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_NE(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                                    if (donor_index, donor_h_set_nac.index, value[0].index) in already_found:
                                        pass
                                    else: 
                                        hb_d1ax_dangling_h = True

                                    if hb_d1ax_dangling_h == True:
                                        self.HB_collect_d1a1_dangling_Ne.append(donor_h_set_nac.index)
                                        #print('dangling H: ', donor_h_set_nac.index); input('enter')

                                    CheckH = HB_collect_check_H_NE(donor_pos_xyz, donor_h_set_nac.position)
                                    if CheckH[0] == True and hb_d1ax_dangling_h == True:
                                        NE_nac_H_d1a1_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== NE D1A1 pdb Window Oindex: %s, nonboned Hindex: %s, Hpos: %s, H chg: %s" %(donor_index, donor_h_set_nac.index, donor_h_set_nac.position[0], self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )) ; input('enter')
                                    if CheckH[1] == True and hb_d1ax_dangling_h == True:
                                        NE_nac_H_d1a1_nonbonded_pdb.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== NE D1A1 pdb Window Oindex: %s, nonboned Hindex: %s, Hpos: %s, H chg: %s" %(donor_index, donor_h_set_nac.index, donor_h_set_nac.position[0], self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )) ; input('enter')
                                    if CheckH[3] == True and hb_d1ax_dangling_h == True:
                                        NE_nac_H_d1a1_nonbonded_tilt.append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                        #print("=== NE D1A1 tilt Window Oindex: %s, nonboned Hindex: %s, Hpos: %s, H chg: %s" %(donor_index, donor_h_set_nac.index, donor_h_set_nac.position[0], self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index] )) ; input('enter')

                                    hb_d1ax_dangling_h = False





                            #    for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                            #        #count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                            #        #count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                            #        self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                        else:
                            #print("D1 No HBs accepted: %s -- %s, %s" %(key.index, key.position[0], donor_pos))
                            ndx_HB_n = 1; ndx_HB_m = 0
                            count_array_NE[ndx_HB_n][ndx_HB_m] += 1
                            count_array_NE_charge_H[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_NE_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])

                            "Collect H D1A0"
                            hb_d1ax_dangling_h = False
                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                #print("*** (Ne) D1A0 Oindex: %s, Hindex: %s, Opos: %s, Hpos: %s" %(donor_index, donor_h_set_nac.index, donor_pos_xyz[0], donor_h_set_nac.position[0])); input('enter')

                                self.HB_collect_d1ax_allH_Ne.append(donor_h_set_nac.index)
                                self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_NE(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                                if (donor_index, donor_h_set_nac.index, value[0].index) in already_found:
                                    pass
                                else: 
                                    hb_d1ax_dangling_h = True

                                hb_d1ax_dangling_h = False


                            #for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                            #    #count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                            #    #count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                            #    self.count_array_NE_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])

                else:
                    tmp_etc += 0  
                    #print("Else", key.index, donor_pos)

            "Analyze HBs for non-donors: the D0 row"
            "Ids start from 1; index start from 0. thus substrate -1 from i."
            donor_no_HB_dict = {i-1:False for i in s1_ids}
            for key, value in dict_d_a.items():
                donor_no_HB_dict[key.index] = True     # true => donating oxygen

                #if key.index == 221: #and a.index == 179:
                #    for val in range(len(value)):
                #        print("D0 - donor(%s)-acceptors: %s" %(donor_index, value[val].index))
                #        input('enter...')

            # value == 999 -> define an arbitary value for the D0 case in order to differenciate them from other D1/D2 in the _update_charge_H_bond_list_IF function.
            #if donor_index == 509:
            #    print(" donor_index: ", donor_index)
            #    input('enter...')
            value = []
            tmp_i = 0
            tmp_f = 0; tmp_if = 0; tmp_im=0; tmp_bulk=0; tmp_ne=0
            for i in range(len(s1_ids)):
                ndx = s1_ids[i] - 1
                tmp_i += 1
                #print(i, "ndx", s1_ids[i], self._s1[i])
                # If i-th donor self._si[i] does not donate
                if donor_no_HB_dict[ndx] == False:
                    tmp_f +=1
                    d_no_HBs = self._s1[i]                 
                    d_no_HBs_pos = d_no_HBs.position[0]
                    donor_index = d_no_HBs.index       # The loop for donor_index was terminated before this block
                    # Call oxygen position and angle  

                    #print(dict_a_d[ndx])
                    #print("len dict_a_d: nr. that accept H ", dict_a_d[ndx])
                    # If the non Donor is near IF; # if d_no_HBs_pos is in the acceptor dictionary
                    if self.cutoff_IF[0] < d_no_HBs_pos <= self.cutoff_IF[1]:
                        #print("d_no_HBs %s at %s " %(d_no_HBs.index, d_no_HBs.position))
                        #print("Accepts nr.", len(dict_a_d[ndx]), dict_a_d[ndx])
                        tmp_if += 1
                        if ndx in dict_a_d:
                            #print("IF len",len(dict_a_d[ndx]))
                            donor_as_acceptor = dict_a_d[ndx]
                            if len(dict_a_d[ndx]) >= 2:
                                " HB statistics EXT. "
                                #print("\nD0A2, value: %s donor_as_acceptor %s"%(len(value), len(donor_as_acceptor) ))
                                
                                count_Dss, count_Ass, Asn0 = _update_DAss(value, donor_as_acceptor)    # update the number of ss/nn/ns 
                                _update_count_array_IF_EXT_D0A2(count_Ass)              # update the extended HB matrix

                                ndx_HB_n = 2; ndx_HB_m = 2
                                count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                # Call oxygen position and angle  
                                tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                                tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                                self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                        #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                        tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                        tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                        tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                        _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)

                                #print("IF: len dict_a_d 2 or more ")
                            if len(dict_a_d[ndx]) == 1:
                                " HB statistics EXT. "
                                #print("\nD0A1, value: %s donor_as_acceptor %s"%(len(value), len(donor_as_acceptor) ))
                                count_Dss, count_Ass, Asn0 = _update_DAss(value, donor_as_acceptor)    # update the number of ss/nn/ns 
                                _update_count_array_IF_EXT_D0A1(count_Ass)              # update the extended HB matrix

                                ndx_HB_n = 2; ndx_HB_m = 1
                                count_array_IF[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                # Call oxygen position and angle  
                                tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                                tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                                self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                " Collect HB D1Ax "
                                #self.HB_collect_d1ax.append(self.index_to_i(donor_index))                                     # make a list for each frame first and print out
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                    "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                    if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                        #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                        tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                        tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                        tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                        _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)


                                #print("IF: len dict_a_d 1")
                        else:
                            " HB statistics EXT. "
                            #print("\nD0A0, value: %s donor_as_acceptor %s"%(len(value), len(donor_as_acceptor) ))
                            count_Dss, count_Ass, Asn0 = _update_DAss(value, donor_as_acceptor)    # update the number of ss/nn/ns 
                            _update_count_array_IF_EXT_D0A0(count_Ass)              # use the same function as D0A1; differentiated by Asn0 = 'True' at D0A0

                            #print("IF %s not in dict_a_d %s"%(ndx, d_no_HBs_pos))
                            ndx_HB_n = 2; ndx_HB_m = 0
                            count_array_IF[ndx_HB_n][ndx_HB_m] += 1       
                            count_array_IF_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_IF_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_IF_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                            # Call oxygen position and angle  
                            tmp_o_angle = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][1]  # [0]: position (x) 
                            tmp_o_cosTheta = self.oxygen_pos_angle_xaxis_HB_tuple_IF[donor_index][2]  # [0]: position (x) 
                            self.IF_charge_list_O_angle_HBmatrix[ndx_HB_n][ndx_HB_m].append((self.nac_water_pos_charge[self.n_frames-1][1][donor_index], tmp_o_angle, tmp_o_cosTheta))
                            " NAC of water and position(x) "
                            donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                            self.IF_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_IF_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_IF_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_IF_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                #print("*****")
                                _update_charge_H_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                                "Call the H-O angle with h.index w.r.t the HB matrix; call h.index only if h.index includes the Au_chg_list"
                                #input('enter....')
                                if donor_h_set_nac.index in self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF:
                                    #print("dict h.index: %s,  h.pos: %s, angle: %s,  au aver_charge: %s" %( donor_h_set_nac.index, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1],  self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]  )) #, self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[1], self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index[2]]]))  
                                    tmp_h_pos = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][0]
                                    tmp_h_angle = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]
                                    tmp_au_near_h_averchg = self.hydrogen_pos_angle_au_chg_xaxis_HB_tuple_IF[donor_h_set_nac.index][2]
                                    _update_charge_H_pos_angle_bond_chg_au_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_pos, tmp_h_angle, tmp_au_near_h_averchg)

                                "Call the H-O angle with h.index w.r.t the HB matrix; call all h.index"
                                if donor_h_set_nac.index in self.hydrogen_pos_angle_xaxis_HB_tuple_IF: 
                                    tmp_h_angle = self.hydrogen_pos_angle_xaxis_HB_tuple_IF[donor_h_set_nac.index][1]  # [0]: position (x) 
                                    _update_charge_H_pos_angle_bond_list_IF(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found, tmp_h_angle)  
                    # If the non Donor is at IM; # if d_no_HBs_pos is in the acceptor dictionary
                    if d_no_HBs_pos > self.cutoff_IF[1] and d_no_HBs_pos < self.cutoff_BULK[0]:
                        #print("d_no_HBs %s at %s " %(d_no_HBs.index, d_no_HBs.position))
                        #print("Accepts nr.", len(dict_a_d[ndx]), dict_a_d[ndx])
                        tmp_im += 1
                        if ndx in dict_a_d:
                            #print("IM len",len(dict_a_d[ndx]))
                            if len(dict_a_d[ndx]) >= 2:
                                ndx_HB_n = 2; ndx_HB_m = 2
                                count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    #"NAC for hydrogen charge when H is not bridged"
                                    #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                                #print("IM: len dict_a_d 2")
                            if len(dict_a_d[ndx]) == 1:
                                ndx_HB_n = 2; ndx_HB_m = 1
                                count_array_IM[ndx_HB_n][ndx_HB_m] += 1
                                count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    #"NAC for hydrogen charge when H is not bridged"
                                    #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                                #print("IM: len dict_a_d 1")
                        else:
                            #print("IM %s not in dict_a_d %s"%(ndx, d_no_HBs_pos))
                            ndx_HB_n = 2; ndx_HB_m = 0
                            count_array_IM[ndx_HB_n][ndx_HB_m] += 1  
                            count_array_IM_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_IM_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_IM_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_IM_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                #"NAC for hydrogen charge when H is not bridged"
                                #_update_charge_H_bond_list_IM(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            #print("IM: len dict_a_d 0")  
                                
                    # If the non Donor is at BULK; # if d_no_HBs_pos is in the acceptor dictionary
                    if d_no_HBs_pos >= self.cutoff_BULK[0] and d_no_HBs_pos <= self.cutoff_BULK[1]:
                        #print("%s D0" %(d_no_HBs.index))
                        #print("Accepts nr.", len(dict_a_d[ndx]), dict_a_d[ndx])
                        tmp_bulk += 1
                        if ndx in dict_a_d:
                            donor_as_acceptor = dict_a_d[ndx]
                            #print("BULK len",len(dict_a_d[ndx]))
                            if len(dict_a_d[ndx]) >= 3:
                                #print("   acceptor (%s)  donor (%s)" %(d_no_HBs.index, self._get_index(donor_as_acceptor)))
                                ndx_HB_n = 3; ndx_HB_m = 3
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            elif len(dict_a_d[ndx]) == 2:
                                #print("   acceptor (%s)  donor (%s)" %(d_no_HBs.index, self._get_index(donor_as_acceptor)))
                                ndx_HB_n = 3; ndx_HB_m = 2
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])

                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)

                            elif len(dict_a_d[ndx]) == 1:
                                #print("   acceptor (%s)  donor (%s)" %(d_no_HBs.index, self._get_index(donor_as_acceptor)))
                                ndx_HB_n = 3; ndx_HB_m = 1
                                count_array_BULK[ndx_HB_n][ndx_HB_m] += 1
                                count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                                count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                                self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])
                                " NAC of water and position(x) "
                                donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                                self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                                for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                    count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                    count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                    self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                    "NAC for hydrogen charge when H is not bridged"
                                    _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)
                               
                        else:
                            #print("BULK %s not in dict_a_d %s"%(ndx, d_no_HBs_pos))
                            ndx_HB_n = 3; ndx_HB_m = 0
                            count_array_BULK[ndx_HB_n][ndx_HB_m] += 1 
                            count_array_BULK_charge_O[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_index]
                            count_array_BULK_charge_O_count[ndx_HB_n][ndx_HB_m] += 1
                            self.count_array_BULK_charge_O_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_index])

                            " NAC of water and position(x) "
                            donor_pos_netWaterchg = self.oxygen_pos_netWaterchg[donor_index]
                            self.BULK_donor_pos_netWaterchg_HBmatrix[ndx_HB_n][ndx_HB_m].append(donor_pos_netWaterchg)

                            for donor_h_set_nac in self._s1_donors_h[self.index_to_i(donor_index)]:
                                count_array_BULK_charge_H[ndx_HB_n][ndx_HB_m] += self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index]
                                count_array_BULK_charge_H_count[ndx_HB_n][ndx_HB_m] += 1 
                                self.count_array_BULK_charge_H_list[ndx_HB_n][ndx_HB_m].append(self.nac_water_pos_charge[self.n_frames-1][1][donor_h_set_nac.index])
                                "NAC for hydrogen charge when H is not bridged"
                                _update_charge_H_bond_list_BULK(ndx_HB_n, ndx_HB_m, value, donor_index, donor_h_set_nac, already_found)                              

                    # If the non Donor is at NE; # if d_no_HBs_pos is in the acceptor dictionary
                    if d_no_HBs_pos > self.cutoff_BULK[1]:
                        #print("d_no_HBs %s at %s " %(d_no_HBs.index, d_no_HBs.position))
                        #print("Accepts nr.", len(dict_a_d[ndx]), dict_a_d[ndx])
                        tmp_ne += 1
                        if ndx in dict_a_d:
                            #print("NE len",len(dict_a_d[ndx]))
                            if len(dict_a_d[ndx]) >= 2:
                                ndx_HB_n = 2; ndx_HB_m = 2
                                count_array_NE[2][2] += 1
                                count_array_NE_charge_H[2][2] += 1
                            if len(dict_a_d[ndx]) == 1:
                                count_array_NE[2][1] += 1
                                count_array_NE_charge_H[2][1] += 1
                        else:
                            #print("NE %s not in dict_a_d %s"%(ndx, d_no_HBs_pos))
                            count_array_NE[2][0] += 1
                            count_array_NE_charge_H[2][0] += 1
            
            t2 = time.time()               


        print("=== End of ts loop ===") 

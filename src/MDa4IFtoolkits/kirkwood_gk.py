import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis  import distances
import os
import sys

class Kirkwood_Gk:
    def __init__(self, u):
        print('u: ', u)#;input('enter')
        self.chg_O = -0.8476; self.chg_H = abs(self.chg_O)/2
        self.mu_aver = 0
        self.eA_to_Debye = 1./0.2081943
        self.dot_mu_1_2 = 0
        self.count = 0
    def angle_between(v1, v2):
        cosTheta = v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        return cosTheta

    def _get_pos_dim(self, pos, _dim):
        if _dim == 'xyz':
            d_position = pos
        elif _dim == 'x':
            d_position = pos 
            d_position[1] = 0.0; d_position[2] = 0.0
        elif _dim == 'yz':
            d_position = pos 
            d_position[0] = 0.0 

        return d_position

    def _get_mu_1_mu_2(self, key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i):
        mu_1 = 0
        d =_s1[dict_Oatom_index_to_i[key]]

        if _dim == 'xyz':
            _s1_positions = _s1.positions
        elif _dim == 'x':
            _s1_positions = _s1.positions
            _s1_positions[:,1] = 0; _s1_positions[:,2] = 0
        elif _dim == 'yz':
            _s1_positions = _s1.positions
            _s1_positions[:,0] = 0.0 

        d_position = self._get_pos_dim(d.position, _dim)
        mu_1 += d_position * self.chg_O * self.eA_to_Debye
        for h in value:
            h_position = self._get_pos_dim(h.position, _dim)
            mu_1 +=  h_position * self.chg_H * self.eA_to_Debye

        mu_1_wc = np.array([0.0, 0.0, 0.0])

        # Find the oxygens around the reference d within the cutoff
        pairs, dist = distances.capped_distance(d.position,
                                                _s1.positions,
                                                max_cutoff=kw_gk_cutoff,
                                                box=_box)    # min_cutoff=0.0

        pairs_arr = np.array(pairs)
        mu_2 = np.array([0.0, 0.0, 0.0]); # dtype=float
        mu_2_wc = np.array([0.0, 0.0, 0.0]); # dtype=float
        for pairs_i in pairs_arr[:,1]:
            o_position = self._get_pos_dim(_s1[pairs_i].position, _dim)
            mu_2 += o_position * self.chg_O * self.eA_to_Debye

            for h_j in dict_oxygen_with_h_tagged[_s1[pairs_i].index]:
                #print('O: ', _s1[pairs_i].index, 'H: ',  h_j.index, h_j.position)
                h_position = self._get_pos_dim(h_j.position, _dim)
                mu_2 += h_position * self.chg_H * self.eA_to_Debye

        return mu_1, mu_2, mu_1_wc, mu_2_wc



    def _get_mu_1_mu_2_for_layer(self, key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, cutoff):
        #print('_dim: ', _dim, key, value)
        mu_1 = 0
        d =_s1[dict_Oatom_index_to_i[key]]

        if _dim == 'xyz':
            _s1_positions = _s1.positions
        elif _dim == 'x':
            _s1_positions = _s1.positions
            _s1_positions[:,1] = 0; _s1_positions[:,2] = 0
        elif _dim == 'yz':
            _s1_positions = _s1.positions
            _s1_positions[:,0] = 0.0 

        d_position = self._get_pos_dim(d.position, _dim)
        #print('  d pos: ', d.index, d_position); #input('enter')
        mu_1 += d_position * self.chg_O * self.eA_to_Debye
        for h in value:
            h_position = self._get_pos_dim(h.position, _dim)
            mu_1 +=  h_position * self.chg_H * self.eA_to_Debye

        mu_1_wc = np.array([0,0,0])

        # Find the oxygens around the reference d within the cutoff
        pairs, dist = distances.capped_distance(d.position,
                                                _s1.positions,
                                                max_cutoff=kw_gk_cutoff,
                                                box=_box)    # min_cutoff=0.0

        ndx = np.where( (_s1.positions[:,0] > cutoff[0]) & (_s1.positions[:,0] <= cutoff[1]))

        pairs_arr = np.array(pairs)
        mu_2 = np.array([0.0, 0.0, 0.0]); # dtype=float
        mu_2_wc = np.array([0.0, 0.0, 0.0]); # dtype=float
       
        count_pairs = 0

        for pairs_i in pairs_arr[:,1]:          # It is for only O around O ref.

            if cutoff[0] < _s1[pairs_i].position[0] <= cutoff[1]:

                o_position = self._get_pos_dim(_s1[pairs_i].position, _dim)
                mu_2 += o_position * self.chg_O * self.eA_to_Debye
                for h_j in dict_oxygen_with_h_tagged[_s1[pairs_i].index]:
                    h_position = self._get_pos_dim(h_j.position, _dim)
                    mu_2 += h_position * self.chg_H * self.eA_to_Debye

        return mu_1, mu_2, mu_1_wc, mu_2_wc


    def _get_mu_1_mu_2_for_layer_count(self, key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, cutoff):
        #print('_dim: ', _dim, key, value)
        mu_1 = 0
        d =_s1[dict_Oatom_index_to_i[key]]

        if _dim == 'xyz':
            _s1_positions = _s1.positions
        elif _dim == 'x':
            _s1_positions = _s1.positions
            _s1_positions[:,1] = 0; _s1_positions[:,2] = 0
        elif _dim == 'yz':
            _s1_positions = _s1.positions
            _s1_positions[:,0] = 0.0 

        d_position = self._get_pos_dim(d.position, _dim)
        print('  d pos: ', d.index, d_position); #input('enter')
        mu_1 += d_position * self.chg_O * self.eA_to_Debye
        for h in value:
            h_position = self._get_pos_dim(h.position, _dim)
            mu_1 +=  h_position * self.chg_H * self.eA_to_Debye

        mu_1_wc = np.array([0,0,0])
        pairs, dist = distances.capped_distance(d.position,
                                                _s1.positions,
                                                max_cutoff=kw_gk_cutoff,
                                                box=_box)    # min_cutoff=0.0

        ndx = np.where( (_s1.positions[:,0] > cutoff[0]) & (_s1.positions[:,0] <= cutoff[1]))

        pairs_arr = np.array(pairs)
        mu_2 = np.array([0.0, 0.0, 0.0]); # dtype=float
        mu_2_wc = np.array([0.0, 0.0, 0.0]); # dtype=float
       
        count_pairs = 0

        for pairs_i in pairs_arr[:,1]:          # It is for only O around O ref.
            if cutoff[0] < _s1[pairs_i].position[0] <= cutoff[1]:

                o_position = self._get_pos_dim(_s1[pairs_i].position, _dim)
                mu_2 += o_position * self.chg_O * self.eA_to_Debye
                count_pairs += 1
                for h_j in dict_oxygen_with_h_tagged[_s1[pairs_i].index]:
                    h_position = self._get_pos_dim(h_j.position, _dim)
                    mu_2 += h_position * self.chg_H * self.eA_to_Debye
        return mu_1, mu_2, mu_1_wc, mu_2_wc, count_pairs

    def _single_frame(self, ts, n_frames_i, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box):
        self.count_kw_gk_ndx = 0; count_o = 0
        dot_mu_1_2 = 0; mu_aver = 0
        dot_mu_1_2_if10AA_xyz = 0; mu_if10AA_xyz = 0; self.count_o_if10AA = 0; mu_if10AA_aver_per_frame_xyz = 0; self.dot_mu_1_2_if10AA_per_frame_xyz = 0
        gk_dot_mu_1_2_if10AA_xyz = 0; gk_dot_mu_1_2_if10AA_x=0; gk_dot_mu_1_2_if10AA_yz=0
        gk_dot_mu_1_2_if12AA_xyz = 0; gk_dot_mu_1_2_if12AA_x=0; gk_dot_mu_1_2_if12AA_yz=0

        dot_mu_1_2_if10AA_x = 0; mu_if10AA_x = 0; self.count_o_if10AA = 0; mu_if10AA_aver_per_frame_x = 0; self.dot_mu_1_2_if10AA_per_frame_x = 0
        dot_mu_1_2_if10AA_yz = 0; mu_if10AA_yz = 0; self.count_o_if10AA = 0; mu_if10AA_aver_per_frame_yz = 0; self.dot_mu_1_2_if10AA_per_frame_yz = 0
        dot_mu_1_2_if = 0; mu_if = 0; self.count_o_if = 0;
        dot_mu_1_2_if_x = 0; mu_if_x = 0; dot_mu_1_2_if_yz = 0; mu_if_yz = 0
        dot_mu_1_2_wc_if = 0;mu_wc_if = 0;  
        dot_mu_1_2_wc_if_x = 0; mu_wc_if_x = 0; dot_mu_1_2_wc_if_yz = 0; mu_wc_if_yz = 0

        dot_mu_1_2_bulk = 0; mu_bulk = 0; self.count_o_bulk = 0;
        dot_mu_1_2_bulk_x = 0; mu_bulk_x = 0; dot_mu_1_2_bulk_yz = 0; mu_bulk_yz = 0
        dot_mu_1_2_wc_bulk = 0;mu_wc_bulk = 0;  
        dot_mu_1_2_wc_bulk_x = 0; mu_wc_bulk_x = 0; dot_mu_1_2_wc_bulk_yz = 0; mu_wc_bulk_yz = 0

        dot_mu_1_2_ne = 0; mu_ne = 0; self.count_o_ne = 0;
        dot_mu_1_2_ne_x = 0; mu_ne_x = 0; dot_mu_1_2_ne_yz = 0; mu_ne_yz = 0
        dot_mu_1_2_wc_ne = 0;mu_wc_ne = 0;  
        dot_mu_1_2_wc_ne_x = 0; mu_wc_ne_x = 0; dot_mu_1_2_wc_ne_yz = 0; mu_wc_ne_yz = 0

        for key, value in dict_oxygen_with_h_tagged.items():
            mu_1 = 0
            d =_s1[dict_Oatom_index_to_i[key]]
            
            _dim = 'xyz'
            # *** skip the calculation
            mu_1 = np.array([0,0,0]); mu_2 = np.array([0,0,0]); mu_1_wc = np.array([0,0,0]); mu_2_wc = mu_1_wc
            mu_1, mu_2, mu_1_wc, mu_2_wc = self._get_mu_1_mu_2(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i,  )

            dot_mu_1_2 += np.dot(mu_1, mu_2)      #print('mu1', mu_1, 'mu2 ', mu_2)
            self.mu_aver += np.linalg.norm(mu_1) 
            mu_aver += np.linalg.norm(mu_1)       # mu_aver per frame

            if d.position[0] <= 10:
                _dim = 'xyz'
                mu_1_if10AA, mu_2_if10AA, mu_1_wc_if10AA, mu_2_wc_if10AA = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 10])
                dot_mu_1_2_if10AA_xyz += np.dot(mu_1_if10AA, mu_2_if10AA)
                mu_if10AA_xyz += np.linalg.norm(mu_1_if10AA)

                _dim = 'yz'
                mu_1_if10AA, mu_2_if10AA, mu_1_wc_if10AA, mu_2_wc_if10AA = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 10])
                dot_mu_1_2_if10AA_yz += np.dot(mu_1_if10AA, mu_2_if10AA)
                mu_if10AA_yz += np.linalg.norm(mu_1_if10AA)

                _dim = 'x'
                mu_1_if10AA, mu_2_if10AA, mu_1_wc_if10AA, mu_2_wc_if10AA = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 10])
                dot_mu_1_2_if10AA_x += np.dot(mu_1_if10AA, mu_2_if10AA)
                mu_if10AA_x += np.linalg.norm(mu_1_if10AA)

                self.count_o_if10AA += 1; 

            if d.position[0] <= 12:
                _dim = 'xyz'
                mu_1_if, mu_2_if, mu_1_wc_if, mu_2_wc_if = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 12])

                dot_mu_1_2_if += np.dot(mu_1_if, mu_2_if)
                mu_if += np.linalg.norm(mu_1_if)       
                dot_mu_1_2_wc_if += np.dot(mu_1_wc_if, mu_2_wc_if)  # update WFc dipole G(k) for the xyz dimension
                mu_wc_if += np.linalg.norm(mu_1_wc_if)       

                _dim = 'x'
                mu_1_if, mu_2_if, mu_1_wc_if, mu_2_wc_if = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 12])
                dot_mu_1_2_if_x += np.dot(mu_1_if, mu_2_if)
                mu_if_x += np.linalg.norm(mu_1_if)
                dot_mu_1_2_wc_if_x += np.dot(mu_1_wc_if, mu_2_wc_if)  # update WFc dipole G(k) for the x dimension
                mu_wc_if_x += np.linalg.norm(mu_1_wc_if)

                _dim = 'yz'
                dot_mu_1_2_if_yz += np.dot(mu_1_if, mu_2_if)
                mu_if_yz += np.linalg.norm(mu_1_if)
                dot_mu_1_2_wc_if_yz += np.dot(mu_1_wc_if, mu_2_wc_if)  # update WFc dipole G(k) for the y and z dimension
                mu_wc_if_yz += np.linalg.norm(mu_1_wc_if)
                self.count_o_if += 1; 

            if d.position[0] > 18 and d.position[0] <= 24:
                _dim = 'xyz'
                mu_1_bulk, mu_2_bulk, mu_1_wc_bulk, mu_2_wc_bulk = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [18-kw_gk_cutoff, 24+kw_gk_cutoff])

                dot_mu_1_2_bulk += np.dot(mu_1_bulk, mu_2_bulk)
                mu_bulk += np.linalg.norm(mu_1_bulk)       
                dot_mu_1_2_wc_bulk += np.dot(mu_1_wc_bulk, mu_2_wc_bulk)  # update WFc dipole G(k) for the xyz dimension
                mu_wc_bulk += np.linalg.norm(mu_1_wc_bulk)       

                _dim = 'x'
                dot_mu_1_2_bulk_x += np.dot(mu_1_bulk, mu_2_bulk)
                mu_bulk_x += np.linalg.norm(mu_1_bulk)
                dot_mu_1_2_wc_bulk_x += np.dot(mu_1_wc_bulk, mu_2_wc_bulk)  # update WFc dipole G(k) for the x dimension
                mu_wc_bulk_x += np.linalg.norm(mu_1_wc_bulk)

                _dim = 'yz'
                dot_mu_1_2_bulk_yz += np.dot(mu_1_bulk, mu_2_bulk)
                mu_bulk_yz += np.linalg.norm(mu_1_bulk)
                dot_mu_1_2_wc_bulk_yz += np.dot(mu_1_wc_bulk, mu_2_wc_bulk)  # update WFc dipole G(k) for the y and z dimension
                mu_wc_bulk_yz += np.linalg.norm(mu_1_wc_bulk)

                self.count_o_bulk += 1

            if d.position[0] > 30.0 and d.position[0] <= 36:
                #print('=== NE ===')
                _dim = 'xyz'
                #----------------------------------------------------------- copied from BULK
                mu_1_ne, mu_2_ne, mu_1_wc_ne, mu_2_wc_ne = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [30, 36])

                dot_mu_1_2_ne += np.dot(mu_1_ne, mu_2_ne)
                mu_ne += np.linalg.norm(mu_1_ne)       
                dot_mu_1_2_wc_ne += np.dot(mu_1_wc_ne, mu_2_wc_ne)  
                mu_wc_ne += np.linalg.norm(mu_1_wc_ne)       

                _dim = 'x'
                dot_mu_1_2_ne_x += np.dot(mu_1_ne, mu_2_ne)
                mu_ne_x += np.linalg.norm(mu_1_ne)
                dot_mu_1_2_wc_ne_x += np.dot(mu_1_wc_ne, mu_2_wc_ne)  
                mu_wc_ne_x += np.linalg.norm(mu_1_wc_ne)

                _dim = 'yz'
                dot_mu_1_2_ne_yz += np.dot(mu_1_ne, mu_2_ne)
                mu_ne_yz += np.linalg.norm(mu_1_ne)
                dot_mu_1_2_wc_ne_yz += np.dot(mu_1_wc_ne, mu_2_wc_ne)  
                mu_wc_ne_yz += np.linalg.norm(mu_1_wc_ne)

                self.count_o_ne += 1

            self.count += 1; count_o += 1

        self.count_kw_gk_ndx += 1
        # average per frame
        mu_aver_per_frame = mu_aver / count_o  
        dot_mu_1_2_per_frame = dot_mu_1_2 / (len(_s1) * mu_aver_per_frame**2)

        if self.count_o_if10AA >= 1:
            mu_if10AA_aver_per_frame_xyz = mu_if10AA_xyz/ self.count_o_if10AA
            self.dot_mu_1_2_if10AA_per_frame_xyz = dot_mu_1_2_if10AA_xyz / (self.count_o_if10AA * mu_if10AA_aver_per_frame_xyz**2)

            mu_if10AA_aver_per_frame_x = mu_if10AA_x/ self.count_o_if10AA
            self.dot_mu_1_2_if10AA_per_frame_x = dot_mu_1_2_if10AA_x / (self.count_o_if10AA * mu_if10AA_aver_per_frame_x**2)

            mu_if10AA_aver_per_frame_yz = mu_if10AA_yz/ self.count_o_if10AA
            self.dot_mu_1_2_if10AA_per_frame_yz = dot_mu_1_2_if10AA_yz / (self.count_o_if10AA * mu_if10AA_aver_per_frame_yz**2)

        mu_if_aver_per_frame = mu_if/ self.count_o_if
        mu_bulk_aver_per_frame = mu_bulk/ self.count_o_bulk
        mu_ne_aver_per_frame = mu_ne/ self.count_o_ne


        # averge per frame; newly updated -------------------------------------------------------------
        self.mu_if = mu_if / (self.count_o_if) 
        self.mu_if_x = mu_if_x / (self.count_o_if)
        self.mu_if_yz = mu_if_yz / (self.count_o_if)

        self.dot_mu_1_2 = dot_mu_1_2 / (len(_s1) * mu_aver_per_frame**2)

        # original code
        self.dot_mu_1_2_if_per_frame = dot_mu_1_2_if / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_if    = dot_mu_1_2_if / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_if_x  = dot_mu_1_2_if_x / (self.count_o_if * self.mu_if_x**2)
        self.dot_mu_1_2_if_yz = dot_mu_1_2_if_yz / (self.count_o_if * self.mu_if_yz**2)

        self.mu_wc_if    = mu_wc_if / (self.count_o_if)
        self.mu_wc_if_x  = mu_wc_if_x / (self.count_o_if)
        self.mu_wc_if_yz = mu_wc_if_yz / (self.count_o_if)
        self.dot_mu_1_2_wc_if    =  dot_mu_1_2_wc_if    / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_wc_if_x  =  dot_mu_1_2_wc_if_x  / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_wc_if_yz =  dot_mu_1_2_wc_if_yz / (self.count_o_if * mu_if_aver_per_frame**2)

        self.mu_bulk = mu_bulk / (self.count_o_bulk) 
        self.mu_bulk_x = mu_bulk_x / (self.count_o_bulk)
        self.mu_bulk_yz = mu_bulk_yz / (self.count_o_bulk)

        self.dot_mu_1_2 = dot_mu_1_2 / (len(_s1) * mu_aver_per_frame**2)

        self.dot_mu_1_2_bulk_per_frame = dot_mu_1_2_bulk / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_bulk    = dot_mu_1_2_bulk / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_bulk_x  = dot_mu_1_2_bulk_x / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_bulk_yz = dot_mu_1_2_bulk_yz / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.mu_wc_bulk    = mu_wc_bulk / (self.count_o_bulk)
        self.mu_wc_bulk_x  = mu_wc_bulk_x / (self.count_o_bulk)
        self.mu_wc_bulk_yz = mu_wc_bulk_yz / (self.count_o_bulk)
        self.dot_mu_1_2_wc_bulk    =  dot_mu_1_2_wc_bulk    / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_wc_bulk_x  =  dot_mu_1_2_wc_bulk_x  / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_wc_bulk_yz =  dot_mu_1_2_wc_bulk_yz / (self.count_o_bulk * mu_bulk_aver_per_frame**2)

        self.mu_ne = mu_ne / (self.count_o_ne) 
        self.mu_ne_x = mu_ne_x / (self.count_o_ne)
        self.mu_ne_yz = mu_ne_yz / (self.count_o_ne)

        self.dot_mu_1_2 = dot_mu_1_2 / (len(_s1) * mu_aver_per_frame**2)

        self.dot_mu_1_2_ne_per_frame = dot_mu_1_2_ne / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_ne    = dot_mu_1_2_ne / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_ne_x  = dot_mu_1_2_ne_x / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_ne_yz = dot_mu_1_2_ne_yz / (self.count_o_ne * mu_ne_aver_per_frame**2)

        self.mu_wc_ne    = mu_wc_ne / (self.count_o_ne)
        self.mu_wc_ne_x  = mu_wc_ne_x / (self.count_o_ne)
        self.mu_wc_ne_yz = mu_wc_ne_yz / (self.count_o_ne)
        self.dot_mu_1_2_wc_ne    =  dot_mu_1_2_wc_ne    / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_wc_ne_x  =  dot_mu_1_2_wc_ne_x  / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_wc_ne_yz =  dot_mu_1_2_wc_ne_yz / (self.count_o_ne * mu_ne_aver_per_frame**2)
        #input('enter')


    def run(self, ts, n_frames_i, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box):
        self._single_frame(ts, n_frames_i, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box)

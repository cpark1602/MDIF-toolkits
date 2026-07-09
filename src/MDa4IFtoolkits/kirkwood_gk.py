import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import os
import sys

class Kirkwood_Gk:
    """
    Calculates the Kirkwood dipole orientation correlation factor (Gk) for water molecules 
    globally and across spatially slabbed sub-layers (e.g., interface vs bulk regions).
    Supports spatial decomposition into complete 3D (xyz), strictly normal (x), or lateral (yz) vector projections.
    """
    def __init__(self, u):
        print('u: ', u)
        
        # --- Topology Charge & Conversion Factors (e.g., SPC/E or TIP3P water parameters) ---
        self.chg_O = -0.8476                     # Partial charge of Oxygen (e.g., SPC/E model)
        self.chg_H = abs(self.chg_O) / 2         # Neutralizing charge per Hydrogen
        self.mu_aver = 0                         # Accumulator for running global average dipole norm
        self.eA_to_Debye = 1. / 0.2081943        # Conversion factor from e*Angstrom to Debye units
        self.dot_mu_1_2 = 0                      # Combined dipole dot product across iterations
        self.count = 0                           # Absolute molecule counter evaluated across trajectory

    def angle_between(v1, v2):
        """Calculates the cosine of the angle between two vectors."""
        cosTheta = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cosTheta

    def _get_pos_dim(self, pos, _dim):
        """
        Projects a 3D coordinate array onto specific spatial dimensions 
        by zeroing out unwanted axis components.
        """
        # Make a copy to avoid mutating the original underlying atom group array
        d_position = np.array(pos, copy=True)
        
        if _dim == 'xyz':
            pass  # Retain original full 3D position
        elif _dim == 'x':
            d_position[1] = 0.0; d_position[2] = 0.0  # Isolate normal vector components (X-axis)
        elif _dim == 'yz':
            d_position[0] = 0.0                       # Isolate lateral vector components (YZ-plane)

        return d_position

    def _get_mu_1_mu_2(self, key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i):
        """
        Computes the target molecular dipole vector (mu_1) alongside the net accumulated 
        dipole moment (mu_2) of all surrounding neighbor molecules within a defined spherical cutoff radius.
        """
        mu_1 = 0
        d = _s1[dict_Oatom_index_to_i[key]]  # Target reference central Oxygen

        # Apply dimensional projection filters to standard array matrix lookups
        if _dim == 'xyz':
            _s1_positions = _s1.positions
        elif _dim == 'x':
            _s1_positions = _s1.positions.copy()
            _s1_positions[:, 1] = 0; _s1_positions[:, 2] = 0
        elif _dim == 'yz':
            _s1_positions = _s1.positions.copy()
            _s1_positions[:, 0] = 0.0 

        # --- Compute Reference Molecule Dipole (mu_1) ---
        d_position = self._get_pos_dim(d.position, _dim)
        mu_1 += d_position * self.chg_O * self.eA_to_Debye
        for h in value:
            h_position = self._get_pos_dim(h.position, _dim)
            mu_1 += h_position * self.chg_H * self.eA_to_Debye

        # Unused placeholder for Wannier/WFC-based dipole calculations
        mu_1_wc = np.array([0.0, 0.0, 0.0])

        # --- Distance Query for Local Coordination Shell ---
        # Find all neighbor Oxygens around reference Oxygen 'd' within the designated spherical shell cutoff
        pairs, dist = distances.capped_distance(d.position,
                                                _s1.positions,
                                                max_cutoff=kw_gk_cutoff,
                                                box=_box)

        pairs_arr = np.array(pairs)
        mu_2 = np.array([0.0, 0.0, 0.0])
        mu_2_wc = np.array([0.0, 0.0, 0.0])
        
        # --- Accumulate Neighbor Dipoles (mu_2) ---
        # Loop over indices of matched neighbor molecules (pairs_arr[:,1] returns the index matches)
        for pairs_i in pairs_arr[:, 1]:
            o_position = self._get_pos_dim(_s1[pairs_i].position, _dim)
            mu_2 += o_position * self.chg_O * self.eA_to_Debye

            # Fetch and process the explicit matched hydrogens linked to this neighbor Oxygen
            for h_j in dict_oxygen_with_h_tagged[_s1[pairs_i].index]:
                h_position = self._get_pos_dim(h_j.position, _dim)
                mu_2 += h_position * self.chg_H * self.eA_to_Debye

        return mu_1, mu_2, mu_1_wc, mu_2_wc

    def _get_mu_1_mu_2_for_layer(self, key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, cutoff):
        """
        Similar to _get_mu_1_mu_2, but additionally filters neighbor molecules (mu_2)
        by checking if they fall strictly within an auxiliary spatial boundary layer along the X-axis.
        """
        mu_1 = 0
        d = _s1[dict_Oatom_index_to_i[key]]

        if _dim == 'xyz':
            _s1_positions = _s1.positions
        elif _dim == 'x':
            _s1_positions = _s1.positions.copy()
            _s1_positions[:, 1] = 0; _s1_positions[:, 2] = 0
        elif _dim == 'yz':
            _s1_positions = _s1.positions.copy()
            _s1_positions[:, 0] = 0.0 

        # --- Compute Reference Dipole ---
        d_position = self._get_pos_dim(d.position, _dim)
        mu_1 += d_position * self.chg_O * self.eA_to_Debye
        for h in value:
            h_position = self._get_pos_dim(h.position, _dim)
            mu_1 += h_position * self.chg_H * self.eA_to_Debye

        mu_1_wc = np.array([0, 0, 0])

        # --- Neighbor Query ---
        pairs, dist = distances.capped_distance(d.position,
                                                _s1.positions,
                                                max_cutoff=kw_gk_cutoff,
                                                box=_box)

        # Vectorized check array for debugging or indexing layer components
        ndx = np.where((_s1.positions[:, 0] > cutoff[0]) & (_s1.positions[:, 0] <= cutoff[1]))

        pairs_arr = np.array(pairs)
        mu_2 = np.array([0.0, 0.0, 0.0])
        mu_2_wc = np.array([0.0, 0.0, 0.0])
       
        # --- Accumulate Layer-Bounded Neighbor Dipoles ---
        for pairs_i in pairs_arr[:, 1]:
            # Spatial restriction check: evaluate neighbor position against layer cutoff along X
            if cutoff[0] < _s1[pairs_i].position[0] <= cutoff[1]:
                o_position = self._get_pos_dim(_s1[pairs_i].position, _dim)
                mu_2 += o_position * self.chg_O * self.eA_to_Debye
                for h_j in dict_oxygen_with_h_tagged[_s1[pairs_i].index]:
                    h_position = self._get_pos_dim(h_j.position, _dim)
                    mu_2 += h_position * self.chg_H * self.eA_to_Debye

        return mu_1, mu_2, mu_1_wc, mu_2_wc

    def _get_mu_1_mu_2_for_layer_count(self, key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, cutoff):
        """
        Extends layer evaluation logic by tracking and returning the total integer number 
        of valid neighbors matched within the slab limits (`count_pairs`).
        """
        mu_1 = 0
        d = _s1[dict_Oatom_index_to_i[key]]

        if _dim == 'xyz':
            _s1_positions = _s1.positions
        elif _dim == 'x':
            _s1_positions = _s1.positions.copy()
            _s1_positions[:, 1] = 0; _s1_positions[:, 2] = 0
        elif _dim == 'yz':
            _s1_positions = _s1.positions.copy()
            _s1_positions[:, 0] = 0.0 

        d_position = self._get_pos_dim(d.position, _dim)
        print('  d pos: ', d.index, d_position)
        mu_1 += d_position * self.chg_O * self.eA_to_Debye
        for h in value:
            h_position = self._get_pos_dim(h.position, _dim)
            mu_1 += h_position * self.chg_H * self.eA_to_Debye

        mu_1_wc = np.array([0, 0, 0])
        pairs, dist = distances.capped_distance(d.position,
                                                _s1.positions,
                                                max_cutoff=kw_gk_cutoff,
                                                box=_box)

        ndx = np.where((_s1.positions[:, 0] > cutoff[0]) & (_s1.positions[:, 0] <= cutoff[1]))

        pairs_arr = np.array(pairs)
        mu_2 = np.array([0.0, 0.0, 0.0])
        mu_2_wc = np.array([0.0, 0.0, 0.0])
       
        count_pairs = 0
        for pairs_i in pairs_arr[:, 1]:
            if cutoff[0] < _s1[pairs_i].position[0] <= cutoff[1]:
                o_position = self._get_pos_dim(_s1[pairs_i].position, _dim)
                mu_2 += o_position * self.chg_O * self.eA_to_Debye
                count_pairs += 1  # Increment matched coordinate counter
                for h_j in dict_oxygen_with_h_tagged[_s1[pairs_i].index]:
                    h_position = self._get_pos_dim(h_j.position, _dim)
                    mu_2 += h_position * self.chg_H * self.eA_to_Debye
                    
        return mu_1, mu_2, mu_1_wc, mu_2_wc, count_pairs

    def _single_frame(self, ts, n_frames_i, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box):
        """
        Evaluates a singular MD frame by iterating through all tracked molecules.
        Calculates dot products across dimensions, aggregates layer statistics, 
        and normalizes cumulative Kirkwood factor tracking registers.
        """
        # =====================================================================
        # Register Initialization (Counters, Normalization Registers, Sub-layers)
        # =====================================================================
        self.count_kw_gk_ndx = 0; count_o = 0
        dot_mu_1_2 = 0; mu_aver = 0
        
        # --- Interface (IF) Layer Definitions ---
        dot_mu_1_2_if10AA_xyz = 0; mu_if10AA_xyz = 0; self.count_o_if10AA = 0; mu_if10AA_aver_per_frame_xyz = 0; self.dot_mu_1_2_if10AA_per_frame_xyz = 0
        gk_dot_mu_1_2_if10AA_xyz = 0; gk_dot_mu_1_2_if10AA_x = 0; gk_dot_mu_1_2_if10AA_yz = 0
        gk_dot_mu_1_2_if12AA_xyz = 0; gk_dot_mu_1_2_if12AA_x = 0; gk_dot_mu_1_2_if12AA_yz = 0

        dot_mu_1_2_if10AA_x = 0; mu_if10AA_x = 0; mu_if10AA_aver_per_frame_x = 0; self.dot_mu_1_2_if10AA_per_frame_x = 0
        dot_mu_1_2_if10AA_yz = 0; mu_if10AA_yz = 0; mu_if10AA_aver_per_frame_yz = 0; self.dot_mu_1_2_if10AA_per_frame_yz = 0
        
        dot_mu_1_2_if = 0; mu_if = 0; self.count_o_if = 0;
        dot_mu_1_2_if_x = 0; mu_if_x = 0; dot_mu_1_2_if_yz = 0; mu_if_yz = 0
        dot_mu_1_2_wc_if = 0; mu_wc_if = 0;  
        dot_mu_1_2_wc_if_x = 0; mu_wc_if_x = 0; dot_mu_1_2_wc_if_yz = 0; mu_wc_if_yz = 0

        # --- Bulk Layer Definitions ---
        dot_mu_1_2_bulk = 0; mu_bulk = 0; self.count_o_bulk = 0;
        dot_mu_1_2_bulk_x = 0; mu_bulk_x = 0; dot_mu_1_2_bulk_yz = 0; mu_bulk_yz = 0
        dot_mu_1_2_wc_bulk = 0; mu_wc_bulk = 0;  
        dot_mu_1_2_wc_bulk_x = 0; mu_wc_bulk_x = 0; dot_mu_1_2_wc_bulk_yz = 0; mu_wc_bulk_yz = 0

        # --- Far-End / Neighboring (NE) Layer Definitions ---
        dot_mu_1_2_ne = 0; mu_ne = 0; self.count_o_ne = 0;
        dot_mu_1_2_ne_x = 0; mu_ne_x = 0; dot_mu_1_2_ne_yz = 0; mu_ne_yz = 0
        dot_mu_1_2_wc_ne = 0; mu_wc_ne = 0;  
        dot_mu_1_2_wc_ne_x = 0; mu_wc_ne_x = 0; dot_mu_1_2_wc_ne_yz = 0; mu_wc_ne_yz = 0

        # =====================================================================
        # Main Processing Loop Over Mapped Molecule Groups
        # =====================================================================
        for key, value in dict_oxygen_with_h_tagged.items():
            mu_1 = 0
            d = _s1[dict_Oatom_index_to_i[key]]
            
            # Global analysis using complete 3D vectors (xyz)
            _dim = 'xyz'
            mu_1 = np.array([0,0,0]); mu_2 = np.array([0,0,0]); mu_1_wc = np.array([0,0,0]); mu_2_wc = mu_1_wc
            
            # Fetch local dipoles
            mu_1, mu_2, mu_1_wc, mu_2_wc = self._get_mu_1_mu_2(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i)

            # Accumulate global dot product interactions and scale profiles
            dot_mu_1_2 += np.dot(mu_1, mu_2)      
            self.mu_aver += np.linalg.norm(mu_1) 
            mu_aver += np.linalg.norm(mu_1)       

            # -----------------------------------------------------------------
            # Sub-Layer Logic: Interface Region (0 to 10 Angstroms)
            # -----------------------------------------------------------------
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

                self.count_o_if10AA += 1

            # -----------------------------------------------------------------
            # Sub-Layer Logic: Extended Interface Region (0 to 12 Angstroms)
            # -----------------------------------------------------------------
            if d.position[0] <= 12:
                _dim = 'xyz'
                mu_1_if, mu_2_if, mu_1_wc_if, mu_2_wc_if = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 12])
                dot_mu_1_2_if += np.dot(mu_1_if, mu_2_if)
                mu_if += np.linalg.norm(mu_1_if)       
                dot_mu_1_2_wc_if += np.dot(mu_1_wc_if, mu_2_wc_if)  
                mu_wc_if += np.linalg.norm(mu_1_wc_if)       

                _dim = 'x'
                mu_1_if, mu_2_if, mu_1_wc_if, mu_2_wc_if = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 12])
                dot_mu_1_2_if_x += np.dot(mu_1_if, mu_2_if)
                mu_if_x += np.linalg.norm(mu_1_if)
                dot_mu_1_2_wc_if_x += np.dot(mu_1_wc_if, mu_2_wc_if)  
                mu_wc_if_x += np.linalg.norm(mu_1_wc_if)

                _dim = 'yz'
                mu_1_if, mu_2_if, mu_1_wc_if, mu_2_wc_if = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [0, 12])
                dot_mu_1_2_if_yz += np.dot(mu_1_if, mu_2_if)
                mu_if_yz += np.linalg.norm(mu_1_if)
                dot_mu_1_2_wc_if_yz += np.dot(mu_1_wc_if, mu_2_wc_if)  
                mu_wc_if_yz += np.linalg.norm(mu_1_wc_if)
                
                self.count_o_if += 1

            # -----------------------------------------------------------------
            # Sub-Layer Logic: Bulk Solvent Layer (18 to 24 Angstroms)
            # -----------------------------------------------------------------
            if d.position[0] > 18 and d.position[0] <= 24:
                _dim = 'xyz'
                # Note the cutoff boundary inflation using kw_gk_cutoff to capture intersecting neighbor buffers
                mu_1_bulk, mu_2_bulk, mu_1_wc_bulk, mu_2_wc_bulk = self._get_mu_1_mu_2_for_layer(key, value, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box, _dim, ts, n_frames_i, [18-kw_gk_cutoff, 24+kw_gk_cutoff])
                dot_mu_1_2_bulk += np.dot(mu_1_bulk, mu_2_bulk)
                mu_bulk += np.linalg.norm(mu_1_bulk)       
                dot_mu_1_2_wc_bulk += np.dot(mu_1_wc_bulk, mu_2_wc_bulk)  
                mu_wc_bulk += np.linalg.norm(mu_1_wc_bulk)       

                _dim = 'x'
                dot_mu_1_2_bulk_x += np.dot(mu_1_bulk, mu_2_bulk)
                mu_bulk_x += np.linalg.norm(mu_1_bulk)
                dot_mu_1_2_wc_bulk_x += np.dot(mu_1_wc_bulk, mu_2_wc_bulk)  
                mu_wc_bulk_x += np.linalg.norm(mu_1_wc_bulk)

                _dim = 'yz'
                dot_mu_1_2_bulk_yz += np.dot(mu_1_bulk, mu_2_bulk)
                mu_bulk_yz += np.linalg.norm(mu_1_bulk)
                dot_mu_1_2_wc_bulk_yz += np.dot(mu_1_wc_bulk, mu_2_wc_bulk)  
                mu_wc_bulk_yz += np.linalg.norm(mu_1_wc_bulk)

                self.count_o_bulk += 1

            # -----------------------------------------------------------------
            # Sub-Layer Logic: Far-End / Outside Region (30 to 36 Angstroms)
            # -----------------------------------------------------------------
            if d.position[0] > 30.0 and d.position[0] <= 36:
                _dim = 'xyz'
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

        # =====================================================================
        # Final Normalizations & Gk Factor Statistical Evaluation
        # =====================================================================
        self.count_kw_gk_ndx += 1
        
        # Calculate mean frame properties
        mu_aver_per_frame = mu_aver / count_o  
        dot_mu_1_2_per_frame = dot_mu_1_2 / (len(_s1) * mu_aver_per_frame**2)

        # Normalize localized 10Å interface values
        if self.count_o_if10AA >= 1:
            mu_if10AA_aver_per_frame_xyz = mu_if10AA_xyz / self.count_o_if10AA
            self.dot_mu_1_2_if10AA_per_frame_xyz = dot_mu_1_2_if10AA_xyz / (self.count_o_if10AA * mu_if10AA_aver_per_frame_xyz**2)

            mu_if10AA_aver_per_frame_x = mu_if10AA_x / self.count_o_if10AA
            self.dot_mu_1_2_if10AA_per_frame_x = dot_mu_1_2_if10AA_x / (self.count_o_if10AA * mu_if10AA_aver_per_frame_x**2)

            mu_if10AA_aver_per_frame_yz = mu_if10AA_yz / self.count_o_if10AA
            self.dot_mu_1_2_if10AA_per_frame_yz = dot_mu_1_2_if10AA_yz / (self.count_o_if10AA * mu_if10AA_aver_per_frame_yz**2)

        # Compute averages of molecule dipole moments per spatial slab region
        mu_if_aver_per_frame = mu_if / self.count_o_if
        mu_bulk_aver_per_frame = mu_bulk / self.count_o_bulk
        mu_ne_aver_per_frame = mu_ne / self.count_o_ne

        # --- Update Class Output Tracking Arrays: Interface (IF) ---
        self.mu_if = mu_if / (self.count_o_if) 
        self.mu_if_x = mu_if_x / (self.count_o_if)
        self.mu_if_yz = mu_if_yz / (self.count_o_if)

        self.dot_mu_1_2 = dot_mu_1_2 / (len(_s1) * mu_aver_per_frame**2)

        self.dot_mu_1_2_if_per_frame = dot_mu_1_2_if / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_if    = dot_mu_1_2_if / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_if_x  = dot_mu_1_2_if_x / (self.count_o_if * self.mu_if_x**2)
        self.dot_mu_1_2_if_yz = dot_mu_1_2_if_yz / (self.count_o_if * self.mu_if_yz**2)

        # Wannier function corrections (WFC) interface components
        self.mu_wc_if    = mu_wc_if / (self.count_o_if)
        self.mu_wc_if_x  = mu_wc_if_x / (self.count_o_if)
        self.mu_wc_if_yz = mu_wc_if_yz / (self.count_o_if)
        self.dot_mu_1_2_wc_if    = dot_mu_1_2_wc_if    / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_wc_if_x  = dot_mu_1_2_wc_if_x  / (self.count_o_if * mu_if_aver_per_frame**2)
        self.dot_mu_1_2_wc_if_yz = dot_mu_1_2_wc_if_yz / (self.count_o_if * mu_if_aver_per_frame**2)

        # --- Update Class Output Tracking Arrays: Bulk ---
        self.mu_bulk = mu_bulk / (self.count_o_bulk) 
        self.mu_bulk_x = mu_bulk_x / (self.count_o_bulk)
        self.mu_bulk_yz = mu_bulk_yz / (self.count_o_bulk)

        self.dot_mu_1_2_bulk_per_frame = dot_mu_1_2_bulk / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_bulk    = dot_mu_1_2_bulk / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_bulk_x  = dot_mu_1_2_bulk_x / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_bulk_yz = dot_mu_1_2_bulk_yz / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        
        self.mu_wc_bulk    = mu_wc_bulk / (self.count_o_bulk)
        self.mu_wc_bulk_x  = mu_wc_bulk_x / (self.count_o_bulk)
        self.mu_wc_bulk_yz = mu_wc_bulk_yz / (self.count_o_bulk)
        self.dot_mu_1_2_wc_bulk    = dot_mu_1_2_wc_bulk    / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_wc_bulk_x  = dot_mu_1_2_wc_bulk_x  / (self.count_o_bulk * mu_bulk_aver_per_frame**2)
        self.dot_mu_1_2_wc_bulk_yz = dot_mu_1_2_wc_bulk_yz / (self.count_o_bulk * mu_bulk_aver_per_frame**2)

        # --- Update Class Output Tracking Arrays: Far-End (NE) ---
        self.mu_ne = mu_ne / (self.count_o_ne) 
        self.mu_ne_x = mu_ne_x / (self.count_o_ne)
        self.mu_ne_yz = mu_ne_yz / (self.count_o_ne)

        self.dot_mu_1_2_ne_per_frame = dot_mu_1_2_ne / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_ne    = dot_mu_1_2_ne / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_ne_x  = dot_mu_1_2_ne_x / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_ne_yz = dot_mu_1_2_ne_yz / (self.count_o_ne * mu_ne_aver_per_frame**2)

        self.mu_wc_ne    = mu_wc_ne / (self.count_o_ne)
        self.mu_wc_ne_x  = mu_wc_ne_x / (self.count_o_ne)
        self.mu_wc_ne_yz = mu_wc_ne_yz / (self.count_o_ne)
        self.dot_mu_1_2_wc_ne    = dot_mu_1_2_wc_ne    / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_wc_ne_x  = dot_mu_1_2_wc_ne_x  / (self.count_o_ne * mu_ne_aver_per_frame**2)
        self.dot_mu_1_2_wc_ne_yz = dot_mu_1_2_wc_ne_yz / (self.count_o_ne * mu_ne_aver_per_frame**2)

    def run(self, ts, n_frames_i, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box):
        """External call API wrapper targeting execution across isolated time frame iterations."""
        self._single_frame(ts, n_frames_i, _s1, dict_Oatom_index_to_i, dict_oxygen_with_h_tagged, kw_gk_cutoff, _box)

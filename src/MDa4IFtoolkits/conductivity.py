#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time

# ----------------------------------
# Path configuration
# ----------------------------------
r_path = "./results"

class Conductivity:
    """
    A class to compute ionic current density and electrical conductivity 
    from Molecular Dynamics (MD) trajectories (typically using MDAnalysis).
    
    Note: This class calculates drift velocity using ONLY the first and last frames.
    For accurate results, the trajectory coordinates MUST be unwrapped (no PBC jumps).
    """
    def __init__(
        self,
        universe,
        selection1,
        selection2,
        #print_results_path,
        Ex,
        start=None,
        stop=None,
        step=None,
        #**kwargs,
    ):  
        self.u = universe                  # MDAnalysis Universe object
        self.seconds1 = time.time()        # Benchmark start time
        self.t_l = []

        self.total_frame = len(self.u.trajectory)
        self.selection1 = selection1        # Atom selection string for cation/ion1
        self.selection2 = selection2        # Atom selection string for anion/ion2
        self.n_frames = None
        self.start = start
        self.stop = stop
        self.step = step
        self.Ex = Ex                        # Applied External Electric Field along X-axis

    def _compute(self, ion_av_vel_x, density_ion, z_ion):
        """
        Helper method to compute current density (J) and conductivity (sigma) for a specific ion species.
        
        Parameters:
        - ion_av_vel_x: 1D array of average velocities along X-axis for each ion [nm/ns]
        - density_ion: Number density of the ion species [ions/nm^3]
        - z_ion: Valency/charge of the ion (e.g., +1, -1)
        """
        # Calculate mean velocity and standard deviation across all ions of this species
        aver_ion = np.mean(ion_av_vel_x)  
        aver_err_ion = np.std(ion_av_vel_x)  

        vel_ion = aver_ion                     # Drift velocity in [nm/ns]
        J_temp_ion = density_ion * vel_ion     # Flux density [ions / (nm^2 * ns)]
        J_temp_ion_err = density_ion * aver_err_ion

        # Elementary charge constant
        e = 1.602 * 10 ** (-19)
        
        # Convert flux density to Charge Current Density [C / (nm^2 * ns)]
        J_C_nm2ns_ion = J_temp_ion * z_ion * e
        J_C_nm2ns_ion_err = J_temp_ion_err * z_ion * e

        # Convert units from C/(nm^2 * ns) to SI units: C/(m^2 * s) or A/m^2
        # (10^9 nm/m)^2 * (10^9 ns/s) = 10^18 * 10^9 = 10^27 total multiplier
        J_ion = J_C_nm2ns_ion * (10 ** (9)) ** 2 * 10 ** (9)
        J_ion_err = J_C_nm2ns_ion_err * (10 ** (9)) ** 2 * 10 ** (9)

        print("Current density : ", J_ion, " [C/(m^2 s)]")
        print("J_ion/self.Ex, ", J_ion, self.Ex)

        # Compute Conductivity: sigma = J / E
        # Multiplied by 10^(-9) presumably to handle the specific unit scale of Ex
        sigma_ion = J_ion / self.Ex * 10 ** (-9)
        sigma_ion_err = J_ion_err / self.Ex * 10 ** (-9)

        print(
            "Conductivity_ion1 : ",
            sigma_ion,
            "(+/- ",
            sigma_ion_err,
            ") [ampere^2 s^3/(kg m^3)]", # Equivalent to S/m (Siemens per meter)
        )
        return J_ion, J_ion_err, sigma_ion, sigma_ion_err

    def run(self, **kwargs):
        """
        Executes the trajectory analysis over the defined frames.
        """
        # Inline helper functions for statistical error propagation
        def compute_stderror(s, sq, c):
            return np.sqrt((sq / c) - (s * s) / (c * c)) / (np.sqrt(c))

        def aver_err(arr, aver):
            length_arr = len(arr)
            sum_deviation = 0
            for i in range(len(arr)):
                sum_deviation = sum_deviation + (arr[i] - aver) ** 2
            std_deviation = np.sqrt((1.0 / (length_arr - 1)) * sum_deviation)  
            return std_deviation / np.sqrt(length_arr)  # Standard error of the mean

        def err_prop(a, b):
            # Propagation of independent errors (square root of sum of squares)
            return np.sqrt(a**2 + b**2)

        self._timeseries = []
        self.timesteps = []
        
        # Select ion populations using MDAnalysis selection strings
        _s1 = self.u.select_atoms(self.selection1)
        _s2 = self.u.select_atoms(self.selection2)
        self.s1_tot_res = len(_s1.ids)

        #### 1. Analyze First Frame ####
        self.u.trajectory[0]
        t_first_ns = self.u.trajectory.time / 1000 # Convert ps to ns
        
        # Convert coordinates from Angstroms (MDAnalysis default) to nanometers
        ion1_pos_first = np.array(_s1.positions) / 10  
        ion2_pos_first = np.array(_s2.positions) / 10  

        #### 2. Analyze Last Frame ####
        self.u.trajectory[-1]
        t_last_ns = self.u.trajectory.time / 1000 # Convert ps to ns

        # Re-fetch positions at the final frame
        _s1 = self.u.select_atoms(self.selection1)
        ion1_pos_last = np.array(_s1.positions) / 10  
        _s2 = self.u.select_atoms(self.selection2)
        ion2_pos_last = np.array(_s2.positions) / 10  

        #### 3. Calculate Average Velocity [nm/ns] ####
        # Net displacement divided by total elapsed time
        ion1_av_vel = (ion1_pos_last - ion1_pos_first) / (t_last_ns - t_first_ns)
        ion2_av_vel = (ion2_pos_last - ion2_pos_first) / (t_last_ns - t_first_ns)

        # Extract only the X-component (assuming electric field is applied along X)
        ion1_av_vel_x = ion1_av_vel[:, 0]
        ion2_av_vel_x = ion2_av_vel[:, 0]

        #### 4. Compute Number Densities ####
        n_ion1 = len(_s1)  
        n_ion2 = len(_s2)  
        n_tot = n_ion1 + n_ion2
        
        box_l = self.u.dimensions[0] / 10  # Box X-length converted from Angstrom to nm
        
        print("Ex : ", self.Ex)
        print("box_l : ", box_l)
        print("n_ion1 : ", n_ion1, "+ n_ion2 : ", n_ion2, "= n_tot :", n_tot)
        
        # Calculate volumes and densities: (N atoms / Volume in Angstroms^3) * 1000 -> atoms/nm^3
        box_volume_A3 = self.u.dimensions[0] * self.u.dimensions[1] * self.u.dimensions[2]
        density_ion1 = (n_ion1 / box_volume_A3) * 1000  
        density_ion2 = (n_ion2 / box_volume_A3) * 1000
        density_tot = (n_tot / box_volume_A3) * 1000

        #### 5. Define Atomic Charges ####
        z_ion1 = +1.0  # Valency of cation
        z_ion2 = -1.0  # Valency of anion
        print("z1: %s z2: %s" % (z_ion1, z_ion2))

        #### 6. Process Individual Ion Properties ####
        print("-----------------ion1--------------------")
        J_ion1, J_ion1_err, sigma_ion1, sigma_ion1_err = self._compute(
            ion1_av_vel_x, density_ion1, z_ion1
        )

        print("----------------- ion2 --------------------")
        J_ion2, J_ion2_err, sigma_ion2, sigma_ion2_err = self._compute(
            ion2_av_vel_x, density_ion2, z_ion2
        )

        #### 7. Calculate Total System Properties ####
        print("-----------------Total--------------------")
        J_tot = J_ion1 + J_ion2
        
        # Scaled current density representation (typically for plotting format)
        J_tot_myunits = J_tot * 10 ** (-6)
        J_tot_err = err_prop(J_ion1_err, J_ion2_err)
        J_tot_err_myunits = J_tot_err * 10 ** (-6)

        print(
            "Total current density : ",
            J_tot_myunits,
            "(+/- ",
            J_tot_err_myunits,
            ") * 10**(6) [A/(m^2)]",
        )

        # Sum up individual contributions to get total electrical conductivity
        sigma_tot = sigma_ion1 + sigma_ion2
        sigma_tot_err = err_prop(sigma_ion1_err, sigma_ion2_err)
        print(
            "Total conductivity : ",
            sigma_tot,
            "(+/- ",
            sigma_tot_err,
            ") [ampere^2 s^3/(kg m^3)]",
        )

        #### 8. Save Outputs to Disk ####
        # Append results: Field Strength (scaled), Value, Error
        fout = open("total_current_density.dat", "a")
        fout.write("%s %s %s\n" % (self.Ex * 1000, J_tot_myunits, J_tot_err_myunits))
        fout.close()

        fout = open("total_conductivity.dat", "a")
        fout.write("%s %s %s\n" % (self.Ex * 1000, sigma_tot, sigma_tot_err))
        fout.close()

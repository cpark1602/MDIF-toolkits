import os
import numpy as np


class Mass_density:
    """
    Calculates the 1D spatial spatial number density profiles of specified atoms 
    along a chosen Cartesian projection axis (x, y, or z) across simulation frames.
    """
    def __init__(
        self,
        universe,
        box,
        print_results_path,
        pbc,
        bin_size,
        dim,
        start=None,
        stop=None,
        step=None,
    ):
        # --- Structural Input Parameters ---
        self.u = universe
        self.box = box              # Numeric dimensions array, e.g., [Lx, Ly, Lz]
        self.bin_size = bin_size    # Width of individual slicing bins in Angstroms
        self.dim = dim              # Linear axis target for profiling ('x', 'y', or 'z')

        # --- Output Target Generation ---
        self.print_results_path = print_results_path
        if not os.path.exists(self.print_results_path):
            os.makedirs(self.print_results_path)

        # --- Trajectory Slice Window Controls ---
        # Evaluate valid dimensions to verify Period Boundary Conditions status
        self.pbc = pbc and all(self.u.dimensions[:3])
        self.start = start if start is not None else 0
        self.stop = stop if stop is not None else len(self.u.trajectory)
        self.step = step if step is not None else 1
        self.start_end_skip = [self.start, self.stop, self.step]

    def _get_res_ids(self, t, input_ion):
        """
        Queries the topology using atom name strings to return absolute 
        atom index IDs from MDAnalysis.
        """
        ion1 = input_ion
        A1 = t.select_atoms("name " + ion1)
        A1_ids = A1.ids  # Extract MDAnalysis 1-indexed identifier IDs array
        
        # Unused topological properties preserved for structural compatibility
        A1_tot_res = len(A1.residues)
        A1_atoms_len = len(A1.residues[0].atoms)
        return A1_ids

    def _get_number_density(self, t, atom_name, dim, bin_size, boxL, start_end_skip):
        """
        Slices the simulation domain space perpendicularly along an explicit dimension,
        accumulates continuous coordinates into histograms, and maps frame density normalized profiles.
        """
        # Fetch targets IDs array
        atom_index = self._get_res_ids(t, atom_name)

        # =====================================================================
        # Axis Dimensional Configuration & Slice Volume Allocation ($V = \Delta L_i \times L_j \times L_k$)
        # =====================================================================
        if dim == "x":
            dim_ndx = 0
            L = boxL[0]
            n_bins = int(boxL[0] / bin_size)
            volume = bin_size * boxL[1] * boxL[2]  # Volume of individual slab bin along X
        elif dim == "y":
            dim_ndx = 1
            L = boxL[1]
            n_bins = int(boxL[1] / bin_size)
            volume = bin_size * boxL[0] * boxL[2]  # Volume of individual slab bin along Y
        else:
            dim_ndx = 2
            L = boxL[2]
            n_bins = int(boxL[2] / bin_size)
            volume = bin_size * boxL[0] * boxL[1]  # Volume of individual slab bin along Z

        frames = 0
        
        # --- Initialization Block: Extract Baseline from First Frame ---
        for ts in t.trajectory[0:1]:
            locpres = t.coord.positions
            # Subtract 1 from MDAnalysis IDs array to map into 0-indexed position array matrices safely
            atom_pos = locpres[atom_index - 1, dim_ndx]
            hist_atom, hist_edges = np.histogram(atom_pos, bins=n_bins, range=(0, L))
            hist_new_atom = hist_atom.copy()  # Instantiate independent accumulator container

            frames += 1
            
        # --- Main Accumulator Processing Loop Over Chosen Chunks ---
        for ts in t.trajectory[
            start_end_skip[0] : start_end_skip[1] : start_end_skip[2]
        ]:
            locpres = t.coord.positions
            atom_pos = locpres[atom_index - 1, dim_ndx]
            
            # Compute structural frequencies mapping per target frame
            hist_atom, hist_edges = np.histogram(atom_pos, bins=n_bins, range=(0, L))
            hist_new_atom += hist_atom  # Accumulate values into tracking array register
            frames += 1

        # =====================================================================
        # Final Normalization Step
        # =====================================================================
        # Divide particle distribution by slab volume units and absolute frame processing scale counts
        number_density = hist_new_atom / volume / frames
        return hist_edges, number_density

    def _get_densityProfile(self, atom_name):
        """
        API wrapper forwarding class setup parameters directly into the spatial density tracking calculator.
        """
        hist_edges, number_density = self._get_number_density(
            self.u, atom_name, self.dim, self.bin_size, self.box, self.start_end_skip
        )
        return hist_edges, number_density

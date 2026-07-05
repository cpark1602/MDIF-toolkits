import os
import numpy as np


class Mass_density:
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

        self.bin_size = bin_size
        self.u = universe
        self.pbc = pbc
        self.bin_size = bin_size
        self.dim = dim

        self.print_results_path = print_results_path
        if not os.path.exists(self.print_results_path):
            os.makedirs(self.print_results_path)

        self.box = box

        self.pbc = pbc and all(self.u.dimensions[:3])
        self.start = start
        self.stop = stop
        self.step = step
        self.start_end_skip = [start, stop, step]

    def _get_res_ids(self, t, input_ion):
        ion1 = input_ion
        A1 = t.select_atoms("name " + ion1)
        A1_ids = A1.ids
        A1_tot_res = len(A1.residues)
        A1_atoms_len = len(A1.residues[0].atoms)
        return A1_ids

    def _get_number_density(self, t, atom_name, dim, bin_size, boxL, start_end_skip):
        atom_index = self._get_res_ids(t, atom_name)

        if dim == "x":
            dim_ndx = 0
            L = boxL[0]
            n_bins = int(boxL[0] / bin_size)
            volume = bin_size * boxL[1] * boxL[2]
        elif dim == "y":
            dim_ndx = 1
            L = boxL[1]
            n_bins = int(boxL[1] / bin_size)
            volume = bin_size * boxL[0] * boxL[2]
        else:
            dim_ndx = 2
            L = boxL[2]
            n_bins = int(boxL[2] / bin_size)
            volume = bin_size * boxL[0] * boxL[1]

        frames = 0
        # Initiate the histogram
        for ts in t.trajectory[0:1]:
            locpres = t.coord.positions
            atom_pos = locpres[atom_index - 1, dim_ndx]
            hist_atom, hist_edges = np.histogram(atom_pos, bins=n_bins, range=(0, L))
            hist_new_atom = hist_atom

            frames += 1
        # run over frames
        for ts in t.trajectory[
            start_end_skip[0] : start_end_skip[1] : start_end_skip[2]
        ]:
            locpres = t.coord.positions
            atom_pos = locpres[atom_index - 1, dim_ndx]
            hist_atom, hist_edges = np.histogram(atom_pos, bins=n_bins, range=(0, L))
            hist_new_atom += hist_atom
            frames += 1

        # final number_density_O
        number_density = hist_new_atom / volume / frames
        return hist_edges, number_density

    # def _get_densityProfile(self, t, atom_name, start_end_skip, boxL, dim):
    def _get_densityProfile(self, atom_name):

        hist_edges, number_density = self._get_number_density(
            self.u, atom_name, self.dim, self.bin_size, self.box, self.start_end_skip
        )
        return hist_edges, number_density

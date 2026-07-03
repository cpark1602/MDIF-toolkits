#!/usr/bin/env python

import MDAnalysis as mda
import numpy as np
import os

import warnings

warnings.filterwarnings(action="once")

import warnings
import logging

logger = logging.getLogger("MDAnalysis.analysis.hbonds")

# ----- Trj path
w_path = "./"
# ----- Load trajectory
u_if = mda.Universe("run-pos.pdb", "run-pos.dcd")

print("total nr. of frame: ", len(u_if.trajectory))
tot_frames = len(u_if.trajectory)
boxX = 48.57
boxY = 15.667
boxZ = 15.076
# boxX = 23.0000; boxY = 23.0000; boxZ = 22.3404
box = [boxX, boxY, boxZ, 90, 90, 90]
u_if.dimensions = box
"Velesco angle H-O-O 35 degrees"
HBs_criteria_input = "Sho"
start_stop_step = [0, 3, 1]

chemisorbed_cutoff_O = 10.5
chemisorbed_cutoff_H = 9.85
# path_results='./results/atomic_charge/'   # where the NAC results are stored
print_results_path = w_path + "/results/"  # To save the results


# ----- Load ACF
# import acf
# if_q0_nac = acf.ACF(u_if, box, HBs_criteria_input, 'name O', 'name O', print_results_path, cutoff_dist_O_H =3.5, cutoff_dist_donor_acceptor = 3.5, cutoff_IF = [0, 12], cutoff_BULK = [19, 28], angle=35.0, pbc=True, start=start_stop_step[0], stop=start_stop_step[1], step=start_stop_step[2], nac='IF')  #tot_frames

# ----- Run analysis
# if_q0_nac.run()


# ----- Load dipole angle analysis
# import dipole_angles
# pbc=True
# dim='x'
# bin_size = 0.02
# selection1 = 'name O'
# if_q0_nac = dangling_bonds.Dangling_bonds(u_if, box, print_results_path, pbc, bin_size, dim, selection1, cutoff_IF = [0, 12], cutoff_BULK = [19, 28], start=start_stop_step[0], stop=start_stop_step[1], step=start_stop_step[2])
#
## ----- Run analysis
# if_q0_nac.run( start_stop_step[0], start_stop_step[1], start_stop_step[2] )


## RDF
# if __name__ == '__main__':
#    import rdf_slab
#    #if not (os.path.exists(tpr_file) and os.path.exists(trj_file)):
#    #    raise FileNotFoundError("Trajectory or Topology path invalid.")
#    IdentityA = "O"
#    IdentityB = "O"
#
#    ag1 = u_if.select_atoms(f'name {IdentityA}')
#    ag2 = u_if.select_atoms(f'name {IdentityB}')
#
#    # Initialize our slab analyzer (Slab bounds set from X=0 to X=4)
#    slab_analyzer = rdf_slab.InterSlabRDF(u_if, box, cutoff_slab=[0, 4], binsize=0.1, exclusion_block=[1, 1])
#
#    print("Starting trajectory analysis loop...")
#    # Loop across frames without manually clearing internal arrays
#    for ts in u_if.trajectory:
#        slab_analyzer._single_frame(ts, ag1, ag2)
#
#    slab_analyzer.conclude()
#
#    # Resolve your query about the first bin:
#    # At r -> 0, Lennard-Jones repulsion means atoms cannot overlap.
#    # Any signal in the first bin is an artifact of bin limits or self-interaction!
#    bins = slab_analyzer.bins
#    rdf_data = slab_analyzer.rdf_slab_global
#
#    # Save arrays cleanly
#    #np.save('rdf_mda_slab_results1.npy', np.array([bins, rdf_data], dtype=object))
#    #
#    ## Output to clean GitHub-friendly dat file
#    #out_name = f"rdf_mda_slab_{IdentityA}-{IdentityB}.dat"
#    #with open(out_name, 'w') as fout:
#    #    for b, r in zip(bins, rdf_data):
#    #        fout.write(f"{b:.4f} {r:.6f}\n")
#    #
#    #print(f"Analysis finished. Data saved to {out_name}")


## Kirkwood g factor
# if __name__ == '__main__':
#    import kirkwood_gk_interface
#
#    pbc=True
#    dim='x'
#    bin_size = 0.02
#    selection1 = 'name O'; selection2 = 'name O'
#    if_q0_nac = kirkwood_gk_interface.Kirkwood_Gk(u_if, box, print_results_path, pbc, bin_size, dim, selection1, selection2, cutoff_IF = [0, 12], cutoff_BULK = [19, 28], start=start_stop_step[0], stop=start_stop_step[1], step=start_stop_step[2])
#
#    # ----- Run analysis
#    #if_q0_nac.run( start_stop_step[0], start_stop_step[1], start_stop_step[2] )
#    if_q0_nac.run()

## Conductivity
# if __name__ == '__main__':
#    import conductivity
#
#    pbc=True
#    Ex = 0.01 # External Field
#    atom1 = 'Li' # Lithium ion
#    atom2 = 'N3' # TFSI ion
#
#
#    u_if_q0_nac = mda.Universe(os.path.join(w_path,trj_file_pdb), os.path.join(w_path,trj_file_trj_1))
#
#    tot_frames = len(u_if_q0_nac.trajectory)
#    start_stop_step = [0, tot_frames, 1]  # q0.0-region2-new/ddec
#    #print_results_path=w_path+'/results-acf-cdot-xyz/'     # To save the results
#    if_q0_nac = conductivity.Conductivity(u_if_q0_nac, 'name '+atom1, 'name '+atom2, print_results_path, Ex,
#                start=start_stop_step[0], stop=start_stop_step[1], step=start_stop_step[2])
#    if_q0_nac.run()


# MSD
if __name__ == "__main__":
    import msd

    trj_file1 = "run-pos.pdb"
    trj_file2 = "run-pos.dcd"

    u = mda.Universe(os.path.join(trj_file1), os.path.join(trj_file2))
    print("total nr. of frame: ", len(u.trajectory))
    tot_frames = len(u.trajectory)
    # skip=int(tot_frames/1000)
    skip = int(1)
    # print('skip: ', skip);#input('enter')
    start_stop_step = [0, tot_frames, skip]  # if xtc gro are loaded.

    u_msd = msd.MSD(
        u,
        select="index 2",
        msd_type="xyz",
        fft=True,
        start=start_stop_step[0],
        stop=start_stop_step[1],
        step=start_stop_step[2],
    )  # tot_frames

    u_msd.run()
    msd = u_msd.timeseries
    nframes = u_msd.n_frames
    timestep = 1  # 0.5
    lagtimes = np.arange(nframes) * timestep * skip

    # np.save('msd_dcd_format', [lagtimes, msd])

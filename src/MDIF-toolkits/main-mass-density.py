#!/usr/bin/env python

import MDAnalysis as mda
import scipy.constants
#import scipy.stats

import warnings

warnings.filterwarnings(action="once")

import warnings
import logging

logger = logging.getLogger("MDAnalysis.analysis.hbonds")

#####----- Trj path -----
w_path = "./"
# from sys import argv
# trj_file1=argv[1]
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
HBs_criteria_input = "Sho"  # Luzar: a rectangule; Sho: Triangle
start_stop_step = [0, 3, 1]  # q0.0-region2-new/ddec

chemisorbed_cutoff_O = 10.5
chemisorbed_cutoff_H = 9.85
# path_results='./results/atomic_charge/'   # where the NAC results are stored
print_results_path = w_path + "/results/"  # To save the results

import mass_density
# if_q0_nac = acf.ACF(u_if, box, HBs_criteria_input, 'name O', 'name O', print_results_path, cutoff_dist_O_H =3.5, cutoff_dist_donor_acceptor = 3.5, cutoff_IF = [0, 12], cutoff_BULK = [19, 28], angle=35.0, pbc=True, start=start_stop_step[0], stop=start_stop_step[1], step=start_stop_step[2], nac='IF')  #tot_frames


dim = "x"
bin_size = 0.02
pbc = True
if_q0_nac = mass_density.Mass_density(
    u_if,
    box,
    print_results_path,
    pbc,
    bin_size,
    dim,
    start=start_stop_step[0],
    stop=start_stop_step[1],
    step=start_stop_step[2],
)

his_edges, number_density_O = if_q0_nac._get_densityProfile("O")
his_edges, number_density_H = if_q0_nac._get_densityProfile("H")
his_edges, number_density_Au = if_q0_nac._get_densityProfile("Au")
his_edges, number_density_Ne = if_q0_nac._get_densityProfile("Ne")


# Mass density (g/L) = N_a / molar mass (kg/mol) * rho(number density)
def number_to_mass(rho_number, molar_mass):
    # #/A^3 * g/mol * mol = g/A^3 * ((10**(10))**3 \AA**3 / m**3) * (1m**3 / 1000 L)  = g / L
    N_a = scipy.constants.physical_constants["Avogadro constant"][0]
    rho_mass = rho_number * molar_mass / N_a * ((10 ** (10)) ** 3) / 1000
    return rho_mass


hydrogen_mass = 1
oxygen_mass = 15.999
Au_mass = 196.96657
Ne_mass = 20.1797
Pt_mass = 195.084


def number_to_mass_g_per_cm3(rho_number, molar_mass):
    # #/A^3 * g/mol * mol = g/A^3 * ((10**(10))**3 \AA**3 / m**3) * (1m**3 / (100cm)^3  = g / L * 1000 L / m^3 / (100)**3
    N_a = scipy.constants.physical_constants["Avogadro constant"][0]
    rho_mass = rho_number * molar_mass / N_a * ((10 ** (10)) ** 3) / (100) ** 3
    return rho_mass


mass_density_O = number_to_mass_g_per_cm3(number_density_O, oxygen_mass)
mass_density_H = number_to_mass_g_per_cm3(number_density_H, hydrogen_mass)
mass_density_Ne = number_to_mass_g_per_cm3(number_density_Ne, Ne_mass)
mass_density_Au = number_to_mass_g_per_cm3(number_density_Au, Au_mass)
mass_density_H2O = mass_density_O + mass_density_H

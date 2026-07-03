import os
import numpy as np
from MDAnalysis import NoDataError
from MDAnalysis.lib import distances
import math as m


class Dangling_bonds:
    def __init__(
        self,
        universe,
        box,
        print_results_path,
        pbc,
        bin_size,
        dim,
        selection1,
        cutoff_IF,
        cutoff_BULK,
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
        self.OH_dist_cutoff = 1.2
        self.cutoff_IF = cutoff_IF
        self.cutoff_BULK = cutoff_BULK

        self.pbc = pbc and all(self.u.dimensions[:3])
        self.start = start
        self.stop = stop
        self.step = step
        self.start_end_skip = [start, stop, step]

        self.selection1 = selection1

        "For angles"
        # Dipole Vector
        self.valdip_IF_xaxis = []
        self.valdip_IM_xaxis = []
        self.valdip_BULK_xaxis = []
        self.valdip_NE_xaxis = []

        self.valdip_IF_yaxis = []
        self.valdip_IM_yaxis = []
        self.valdip_BULK_yaxis = []
        self.valdip_NE_yaxis = []

        self.valdip_IF_zaxis = []
        self.valdip_IM_zaxis = []
        self.valdip_BULK_zaxis = []
        self.valdip_NE_zaxis = []

        self.cosTheta_valdip_IF_xaxis = []
        self.cosTheta_valdip_IM_xaxis = []
        self.cosTheta_valdip_BULK_xaxis = []
        self.cosTheta_valdip_NE_xaxis = []

        self.cosTheta_valdip_IF_yaxis = []
        self.cosTheta_valdip_IM_yaxis = []
        self.cosTheta_valdip_BULK_yaxis = []
        self.cosTheta_valdip_NE_yaxis = []

        self.cosTheta_valdip_IF_zaxis = []
        self.cosTheta_valdip_IM_zaxis = []
        self.cosTheta_valdip_BULK_zaxis = []
        self.cosTheta_valdip_NE_zaxis = []

        # OH Vector
        self.vecOH_IF_xaxis = []
        self.vecOH_IM_xaxis = []
        self.vecOH_BULK_xaxis = []
        self.vecOH_NE_xaxis = []

        self.vecOH_IF_yaxis = []
        self.vecOH_IM_yaxis = []
        self.vecOH_BULK_yaxis = []
        self.vecOH_NE_yaxis = []

        self.vecOH_IF_zaxis = []
        self.vecOH_IM_zaxis = []

        self.vecOH_BULK_zaxis = []
        self.vecOH_NE_zaxis = []

        self.cosTheta_vecOH_IF_xaxis = []
        self.cosTheta_vecOH_IM_xaxis = []
        self.cosTheta_vecOH_BULK_xaxis = []
        self.cosTheta_vecOH_NE_xaxis = []

        self.cosTheta_vecOH_IF_yaxis = []
        self.cosTheta_vecOH_IM_yaxis = []
        self.cosTheta_vecOH_BULK_yaxis = []
        self.cosTheta_vecOH_NE_yaxis = []

        self.cosTheta_vecOH_IF_zaxis = []
        self.cosTheta_vecOH_IM_zaxis = []
        self.cosTheta_vecOH_BULK_zaxis = []
        self.cosTheta_vecOH_NE_zaxis = []

    def _get_bonded_hydrogens_dist(self, atom):
        """Find bonded hydrogens within cutoff to 'atom'.
        Hydrogen bonds are detected by the cutoff;
        The distance from the reference 'atom' is calculated for all hydrogens in the residue
        and only those within a cutoff are kept."""
        try:
            sel_h = atom.residue.atoms.select_atoms(
                "(name H) and around {0:f} index {1!s}".format(
                    self.OH_dist_cutoff, atom.index
                )
            )
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
                self._s1_donors_h[i] = tmp  # fill the dict[i]

    def unit_vector(self, vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)

        cosTheta = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        return cosTheta

    # Note: returns angle in degree and cosine(theta)
    def vector2degree(self, vec, axis):
        if axis == "x":
            axis_ndx = 0
            unit_vec = np.array([1, 0, 0])
        elif axis == "y":
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

        dcoordO = []
        xcoordH = []
        ycoordH = []
        zcoordH = []
        tmp_H_position = []
        h_index_tmp_hb = []
        ohb = 0
        for i in range(len(self._s1_donors_h[ndx])):
            new_H_position = self._s1_donors_h[ndx][i].position[:3]
            h_index_tmp_hb.append(self._s1_donors_h[ndx][i].index)
            xcoordH.append(self._s1_donors_h[ndx][i].position[0])
            ycoordH.append(self._s1_donors_h[ndx][i].position[1])
            zcoordH.append(self._s1_donors_h[ndx][i].position[2])

            ohbond_dist = distances.calc_bonds(
                d.position, self._s1_donors_h[ndx][i].position, box=None
            )  # self.box
            if ohbond_dist > 2:
                ohb = 1

        coordO = d.position
        # no change in the x-direction
        dcoordO.append(float(0.0))
        bcell = self.box[1]
        ccell = self.box[2]
        # check the y direction
        if coordO[1] >= bcell:
            dcoordO.append(m.floor(coordO[1] / bcell) * bcell)
        elif coordO[1] < float(0):
            dcoordO.append(-m.ceil(abs(coordO[1]) / bcell) * bcell)
        else:
            dcoordO.append(float(0.0))
        # check the z direction
        if coordO[2] > ccell:
            dcoordO.append(m.floor(coordO[2] / ccell) * ccell)
        elif coordO[2] <= float(0):
            dcoordO.append(-m.ceil(abs(coordO[2]) / ccell) * ccell)
        else:
            dcoordO.append(float(0.0))

        xcoordHnew = xcoordH[0] - dcoordO[0]
        ycoordHnew = ycoordH[0] - dcoordO[1]
        zcoordHnew = zcoordH[0] - dcoordO[2]
        tmp_H_position.append(np.array([xcoordHnew, ycoordHnew, zcoordHnew]))

        string1 = (
            "  "
            + "H"
            + " "
            + str("{:20.10f}".format(xcoordHnew))
            + str("{:20.10f}".format(ycoordHnew))
            + str("{:20.10f}".format(zcoordHnew))
            + "\n"
        )

        xcoordHnew = xcoordH[1] - dcoordO[0]
        ycoordHnew = ycoordH[1] - dcoordO[1]
        zcoordHnew = zcoordH[1] - dcoordO[2]
        # print('new H2 pos: ', xcoordHnew, ycoordHnew, zcoordHnew)
        tmp_H_position.append(np.array([xcoordHnew, ycoordHnew, zcoordHnew]))

        string2 = (
            "  "
            + "H"
            + " "
            + str("{:20.10f}".format(xcoordHnew))
            + str("{:20.10f}".format(ycoordHnew))
            + str("{:20.10f}".format(zcoordHnew))
            + "\n"
        )

        coordO[:] = [float(p - q) for (p, q) in zip(coordO, dcoordO)]
        string3 = (
            "  "
            + "O"
            + " "
            + str("{:20.10f}".format(coordO[0]))
            + str("{:20.10f}".format(coordO[1]))
            + str("{:20.10f}".format(coordO[2]))
            + "\n"
        )

        if ohb == 1:
            for hb_iii in range(len(tmp_H_position)):
                dcoordH_a = []
                ohbond_dist = distances.calc_bonds(
                    coordO, tmp_H_position[hb_iii], box=None
                )  # self.box
                if ohbond_dist > 2.0:
                    # print('ohbond_dist', ohbond_dist, tmp_H_position[hb_iii])
                    ohbond_dist_pbc = distances.calc_bonds(
                        coordO, tmp_H_position[hb_iii], box=self.box
                    )  #
                    dcoordH_a.append(float(0.0))
                    bcell = self.box[1]
                    ccell = self.box[2]
                    # check the y direction
                    if tmp_H_position[hb_iii][1] >= bcell:
                        dcoordH_a.append(
                            m.floor(tmp_H_position[hb_iii][1] / bcell) * bcell
                        )
                    elif tmp_H_position[hb_iii][1] < float(0):
                        dcoordH_a.append(
                            -m.ceil(abs(tmp_H_position[hb_iii][1]) / bcell) * bcell
                        )
                    else:
                        dcoordH_a.append(float(0.0))
                    # check the z direction
                    if tmp_H_position[hb_iii][2] > ccell:
                        dcoordH_a.append(
                            m.floor(tmp_H_position[hb_iii][2] / ccell) * ccell
                        )
                    elif tmp_H_position[hb_iii][2] <= float(0):
                        dcoordH_a.append(
                            -m.ceil(abs(tmp_H_position[hb_iii][2]) / ccell) * ccell
                        )
                    else:
                        dcoordH_a.append(float(0.0))

                    xcoordHnew = tmp_H_position[hb_iii][0] - dcoordH_a[0]
                    ycoordHnew = tmp_H_position[hb_iii][1] - dcoordH_a[1]
                    zcoordHnew = tmp_H_position[hb_iii][2] - dcoordH_a[2]
                    newH_pos = np.array([xcoordHnew, ycoordHnew, zcoordHnew])
                    tmp_H_position[hb_iii] = newH_pos
                    ohbond_dist_new = distances.calc_bonds(coordO, newH_pos, box=None)
                    delta_ohbond_dist = abs(ohbond_dist_new - ohbond_dist_pbc)
                    self.delta_ohbond_dist_list.append(delta_ohbond_dist)
                    if delta_ohbond_dist > 0.5:
                        ohbond_dist_tmp = distances.calc_bonds(
                            np.array([0, coordO[1], 0]),
                            np.array([0, newH_pos[1], 0]),
                            box=None,
                        )
                        arbitary_cutoff = 2.0
                        if (
                            coordO[1] <= self.box[1] / 2
                            and newH_pos[1] > self.box[1] / 2
                            and ohbond_dist_tmp > arbitary_cutoff
                        ):
                            newH_pos[1] = self.box[1] - newH_pos[1]
                        if (
                            coordO[1] >= self.box[1] / 2
                            and newH_pos[1] < self.box[1] / 2
                            and ohbond_dist_tmp > arbitary_cutoff
                        ):
                            newH_pos[1] = self.box[1] + newH_pos[1]
                        ohbond_dist_tmp = distances.calc_bonds(
                            np.array([0, 0, coordO[2]]),
                            np.array([0, 0, newH_pos[2]]),
                            box=None,
                        )
                        if (
                            coordO[2] <= self.box[2] / 2
                            and newH_pos[2] > self.box[2] / 2
                            and ohbond_dist_tmp > arbitary_cutoff
                        ):
                            newH_pos[2] = self.box[2] - newH_pos[2]
                        if (
                            coordO[2] >= self.box[2] / 2
                            and newH_pos[2] < self.box[2] / 2
                            and ohbond_dist_tmp > arbitary_cutoff
                        ):
                            newH_pos[2] = self.box[2] + newH_pos[2]
                    ohbond_dist_new = distances.calc_bonds(coordO, newH_pos, box=None)
                    ohbond_dist_pbc = distances.calc_bonds(
                        coordO, newH_pos, box=self.box
                    )  #
                    delta_ohbond_dist = abs(ohbond_dist_new - ohbond_dist_pbc)

        "obtain the OH vectors and dipole vector"
        # OH_Vector_1 = tmp_H_position[0] - d.position   # toward H coordO
        OH_Vector_1 = tmp_H_position[0] - coordO  # toward O
        OH_Vector_2 = tmp_H_position[1] - coordO  # toward H
        # OH_Vector_2 = d.position - tmp_H_position[1]    # toward O

        # print(type(tmp_H_position[0]),  tmp_H_position)
        dipVector = (tmp_H_position[0] + tmp_H_position[1]) * 0.5 - coordO
        # print("[", dipVector[0], ",", dipVector[1], ",", dipVector[2], "], ")
        # self.dipVector_list.append(dipVector)

        tmp = 0
        tmp_f = 0
        tmp_if = 0
        tmp_im = 0
        tmp_bulk = 0
        tmp_ne = 0
        tmp_etc = 0

        angle_dip_xaxis, cosTheta_dip_xaxis = self.vector2degree(dipVector, "x")
        angle_dip_yaxis, cosTheta_dip_yaxis = self.vector2degree(dipVector, "y")
        angle_dip_zaxis, cosTheta_dip_zaxis = self.vector2degree(dipVector, "z")

        angle_OH_1_xaxis, cosTheta_OH_1_xaxis = self.vector2degree(OH_Vector_1, "x")
        angle_OH_1_yaxis, cosTheta_OH_1_yaxis = self.vector2degree(OH_Vector_1, "y")
        angle_OH_1_zaxis, cosTheta_OH_1_zaxis = self.vector2degree(OH_Vector_1, "z")

        angle_OH_2_xaxis, cosTheta_OH_2_xaxis = self.vector2degree(OH_Vector_2, "x")
        angle_OH_2_yaxis, cosTheta_OH_2_yaxis = self.vector2degree(OH_Vector_2, "y")
        angle_OH_2_zaxis, cosTheta_OH_2_zaxis = self.vector2degree(OH_Vector_2, "z")

        # variables for feeOH analysis
        S_tmp = 0
        dd = 0
        nd = 0
        tmp_freeOH = 0
        SD_parallel = 0
        SD_perpendicular = 0
        freeOH_perpendicular = 0
        freeOH_parallel = 0
        tmp_d = 0
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

        # return (nd, SD_perpendicular, SD_parallel, dd)

    def run(self, start, stop, step):

        for ts in self.u.trajectory[start:stop:step]:
            # Select atom groups
            self._s1 = self.u.select_atoms(self.selection1)

            # _update_selection_1() call the function _get_bonded_hydrogens_dist
            # to find the bonded Hydrogens
            self._update_selection_1()

            DD_tmp = 0
            SD_perpendicular_tmp = 0
            SD_parallel_tmp = 0
            ND_tmp = 0
            for i, donor_h_set in self._s1_donors_h.items():
                # i-th donor in the atom roup _s1 (selection1)
                d = self._s1[i]

                "-----Analyze Angle-----"
                # Input: the ids, i, of Oxygen atoms
                # return (ND, SD_perpendicular, SD_parallel, DD)
                self._getCosTheta(i)

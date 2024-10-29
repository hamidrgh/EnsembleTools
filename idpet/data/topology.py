"""
Work with topologies of protein systems.
"""

import mdtraj
import numpy as np


aa_three_letters = ["GLN", "TRP", "GLU", "ARG", "THR",
                    "TYR", "ILE", "PRO", "ALA", "SER",
                    "ASP", "PHE", "GLY", "HIS", "LYS",
                    "LEU", "CYS", "VAL", "ASN", "MET"]

aa_one_to_three_dict = {
    "G": "GLY", "A": "ALA", "L": "LEU", "I": "ILE", "R": "ARG", "K": "LYS",
    "M": "MET", "C": "CYS", "Y": "TYR", "T": "THR", "P": "PRO", "S": "SER",
    "W": "TRP", "D": "ASP", "E": "GLU", "N": "ASN", "Q": "GLN", "F": "PHE",
    "H": "HIS", "V": "VAL", "X": "UNK"
}

def get_ca_topology(sequence: str, bead_name: str = "CA") -> mdtraj.Topology:
    """
    input: amino acid sequence.
    output: a mdtraj topology with one bead per residue.
    """
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for res in sequence:
        res_obj = topology.add_residue(aa_one_to_three_dict[res], chain)
        topology.add_atom(bead_name, mdtraj.core.topology.elem.carbon, res_obj)
    return topology

def slice_traj_to_com(traj: mdtraj.Trajectory) -> mdtraj.Trajectory:
    ha_ids = [a.index for a in traj.topology.atoms if \
              a.residue.name in aa_three_letters and \
              a.element.symbol != "H"]
    ha_traj = traj.atom_slice(ha_ids)
    residues = list(ha_traj.topology.residues)
    com_xyz = np.zeros((ha_traj.xyz.shape[0], len(residues), 3))
    for i, residue_i in enumerate(residues):
        ha_ids_i = [a.index for a in residue_i.atoms]
        masses_i = np.array([a.element.mass for a in residue_i.atoms])
        masses_i = masses_i[None,:,None]
        tot_mass_i = masses_i.sum()
        com_xyz_i = np.sum(ha_traj.xyz[:,ha_ids_i,:]*masses_i, axis=1)/tot_mass_i
        com_xyz[:,i,:] = com_xyz_i
    return mdtraj.Trajectory(
        xyz=com_xyz,
        topology=get_ca_topology(
            sequence="".join([r.code for r in ha_traj.topology.residues])
        ))
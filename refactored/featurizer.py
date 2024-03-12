from abc import ABC, abstractmethod
import mdtraj
import numpy as np

class Featurizer(ABC):
    @abstractmethod
    def featurize(self, traj, get_names=False):
        pass

class CADistFeaturizer(Featurizer):
    def __init__(self, seq_sep, inverse):
        self.seq_sep = seq_sep
        self.inverse = inverse

    def featurize(self, traj, get_names=False):
        ca_ids = traj.topology.select("name == CA")
        atoms = list(traj.topology.atoms)
        # Get all pair of ids.
        pair_ids = []
        names = []
        for i, id_i in enumerate(ca_ids):
            for j, id_j in enumerate(ca_ids):
                if j - i >= self.seq_sep:
                    pair_ids.append([id_i, id_j])
                    if get_names:
                        names.append(
                            f"{repr(atoms[id_i].residue)}-{repr(atoms[id_j].residue)}"
                        )
        # Calculate C-alpha - C-alpha distances.
        ca_dist = mdtraj.compute_distances(traj=traj, atom_pairs=pair_ids)
        if self.inverse:
            ca_dist = 1 / ca_dist
        if get_names:
            return ca_dist, names
        else:
            return ca_dist

class PhiPsiFeaturizer(Featurizer):
    def featurize(self, traj, get_names=False):
        atoms = list(traj.topology.atoms)
        phi_ids, phi = mdtraj.compute_phi(traj)
        psi_ids, psi = mdtraj.compute_psi(traj)
        phi_psi = np.concatenate([phi, psi], axis=1)
        if get_names:
            names = []
            for t in phi_ids:
                names.append(repr(atoms[t[1]].residue) + "-PHI")
            for t in psi_ids:
                names.append(repr(atoms[t[0]].residue) + "-PSI")
            return phi_psi, names
        else:
            return phi_psi

class AAngleFeaturizer(Featurizer):
    def featurize(self, traj, get_names=False):
        # Get all C-alpha indices.
        ca_ids = traj.topology.select("protein and name CB")
        atoms = list(traj.topology.atoms)
        # Get all pair of ids.
        tors_ids = []
        names = []
        for i in range(ca_ids.shape[0] - 3):
            if get_names:
                names.append(
                    "{}:{}:{}:{}".format(
                        atoms[ca_ids[i]],
                        atoms[ca_ids[i + 1]],
                        atoms[ca_ids[i + 2]],
                        atoms[ca_ids[i + 3]],
                    )
                )
            tors_ids.append((ca_ids[i], ca_ids[i + 1], ca_ids[i + 2], ca_ids[i + 3]))
        tors = mdtraj.compute_dihedrals(traj, tors_ids)
        if get_names:
            return tors, names
        else:
            return tors

class FeaturizationFactory:
    @staticmethod
    def get_featurizer(featurization, *args, **kwargs):
        if featurization == "ca_dist":
            return CADistFeaturizer(*args, **kwargs)
        elif featurization == "phi_psi":
            return PhiPsiFeaturizer(*args, **kwargs)
        elif featurization == "a_angle":
            return AAngleFeaturizer(*args, **kwargs)
        else:
            raise NotImplementedError("Unsupported feature extraction method.")

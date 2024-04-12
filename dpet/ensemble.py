import os
import mdtraj
import numpy as np

from dpet.featurization.angles import featurize_a_angle, featurize_phi_psi, featurize_tr_angle
from dpet.featurization.distances import featurize_ca_dist


class Ensemble():
    def __init__(self, ens_code, data_dir) -> None:
        self.ens_code = ens_code
        self.data_dir = data_dir

    def load_trajectory(self):
        pdb_filename = f'{self.ens_code}.pdb'
        pdb_file = os.path.join(self.data_dir, pdb_filename)
        traj_dcd = os.path.join(self.data_dir, f'{self.ens_code}.dcd')
        traj_xtc = os.path.join(self.data_dir, f'{self.ens_code}.xtc')
        traj_top = os.path.join(self.data_dir, f'{self.ens_code}.top.pdb')
        
        ens_dir = os.path.join(self.data_dir, self.ens_code)

        if os.path.exists(traj_dcd) and os.path.exists(traj_top):
            print(f'Trajectory already exists for ensemble {self.ens_code}. Loading trajectory.')
            self.trajectory = mdtraj.load(traj_dcd, top=traj_top)
        elif os.path.exists(traj_xtc) and os.path.exists(traj_top):
            print(f'Trajectory already exists for ensemble {self.ens_code}. Loading trajectory.')
            self.trajectory = mdtraj.load(traj_xtc, top=traj_top)
        elif os.path.exists(pdb_file):
            print(f'Generating trajectory from PDB file: {pdb_file}.')
            self.trajectory = mdtraj.load(pdb_file)
            print(f'Saving trajectory.')
            self.trajectory.save(traj_dcd)
            self.trajectory[0].save(traj_top)
        elif os.path.exists(ens_dir):
            files_in_dir = [f for f in os.listdir(ens_dir) if f.endswith('.pdb')]
            if files_in_dir:
                full_paths = [os.path.join(ens_dir, file) for file in files_in_dir]
                print(f'Generating trajectory from directory: {ens_dir}.')
                self.trajectory = mdtraj.load(full_paths)
                print(f'Saving trajectory.')
                self.trajectory.save(traj_dcd)
                self.trajectory[0].save(traj_top)
            else:
                print(f"No DCD files found in directory: {ens_dir}")
        else:
            print(f"File or directory for ensemble {self.ens_code} doesn't exist.")
            return
        self.original_trajectory = self.trajectory[:]
        
    def check_coarse_grained(self):
        topology = self.trajectory.topology
        atoms = topology.atoms
        self.coarse_grained = all(atom.element.symbol == 'C' for atom in atoms)
        self.atom_selector = "protein" if self.coarse_grained else "protein and name CA"

    def random_sample_trajectory(self, sample_size:int):
        total_frames = len(self.original_trajectory)
        if sample_size > total_frames:
            raise ValueError("Sample size cannot be larger than the total number of frames in the trajectory.")
        random_indices = np.random.choice(total_frames, size=sample_size, replace=False)
        self.trajectory = mdtraj.Trajectory(
            xyz=self.original_trajectory.xyz[random_indices],
            topology=self.original_trajectory.topology)
        
    def extract_features(self, featurization, *args, **kwargs):
        print(f"Performing feature extraction for Ensemble: {self.ens_code}.")
        self.features, self.names = self._featurize(featurization, *args, **kwargs)
        print("Transformed ensemble shape:", self.features.shape)

    def _featurize(self, featurization: str, *args, **kwargs):
        if featurization == "ca_dist":
            return featurize_ca_dist(
                traj=self.trajectory, 
                get_names=True,
                atom_selector=self.atom_selector,
                *args, **kwargs)
        elif featurization == "phi_psi":
            return featurize_phi_psi(
                traj=self.trajectory, 
                get_names=True, 
                *args, **kwargs)
        elif featurization == "a_angle":
            return featurize_a_angle(
                traj=self.trajectory, 
                get_names=True, 
                atom_selector=self.atom_selector,
                *args, **kwargs)
        elif featurization == "tr_omega":
            return featurize_tr_angle(
                traj=self.trajectory,
                type="omega",
                get_names=True,
                *args, **kwargs)
        elif featurization == "tr_phi":
            return featurize_tr_angle(
                traj=self.trajectory,
                type="phi",
                get_names=True,
                *args, **kwargs)
        else:
            raise NotImplementedError("Unsupported feature extraction method.")
        
    def normalize_features(self, mean, std):
        self.features = (self.features - mean) / std
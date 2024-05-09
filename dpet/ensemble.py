import os
import sys
import mdtraj
import numpy as np

from dpet.featurization.angles import featurize_a_angle, featurize_phi_psi, featurize_tr_angle
from dpet.featurization.distances import featurize_ca_dist


class Ensemble():
    """
    Represents a molecular dynamics ensemble.

    Parameters
    ----------
    code : str
        The code identifier of the ensemble.
    data_path : str, optional
        The path to the data file associated with the ensemble. Default is None.
    top_path : str, optional
        The path to the topology file associated with the ensemble. Default is None.
    database : str, optional
        The database from which to download the ensemble. Options are 'ped' and 'atlas'. Default is None.
    chain_id : int, optional
        MDtraj chain identifier used to select a single chain to analyze in case multiple chains are loaded. Default is None.

    Notes
    -----
    - If the database is 'atlas', the ensemble code should be provided as a PDB ID with a chain identifier separated by an underscore. 
    Example: '3a1g_B'.
    - If the database is 'ped', the ensemble code should be in the PED ID format, which consists of a string starting with 'PED' followed by a numeric identifier,
    and 'e' followed by another numeric identifier. Example: 'PED00423e001'.
    - `chain_id` is always assigned in order to the actual chains. For example, chains A, C, D would have `chain_id` 0, 1, 2 respectively.
    """
    def __init__(self, code: str, data_path: str = None, top_path: str = None, database: str = None, chain_id: int = None) -> None:
        self.code = code
        self.data_path = data_path
        self.top_path = top_path
        self.database = database
        self.chain_id = chain_id
    
    def load_trajectory(self, data_dir: str):  
        if not os.path.exists(self.data_path):
            print(f"Data file or directory for ensemble {self.code} doesn't exist.")
            return
        elif self.data_path.endswith('.pdb'):
            print(f"Generating trajectory for {self.code}...")
            self.trajectory = mdtraj.load(self.data_path)
            traj_dcd = os.path.join(data_dir, f'{self.code}.dcd')
            traj_top = os.path.join(data_dir, f'{self.code}.top.pdb')
            self.trajectory.save(traj_dcd)
            self.trajectory[0].save(traj_top)
            print(f"Generated trajectory saved to {data_dir}.")
        elif (self.data_path.endswith('.dcd') or self.data_path.endswith('.xtc')) and os.path.exists(self.top_path):
            print(f"Loading trajectory for {self.code}...")
            self.trajectory = mdtraj.load(self.data_path, top=(self.top_path))
        elif os.path.isdir(self.data_path):
            files_in_dir = [f for f in os.listdir(self.data_path) if f.endswith('.pdb')]
            if files_in_dir:
                print(f"Generating trajectory for {self.code}...")
                full_paths = [os.path.join(self.data_path, file) for file in files_in_dir]
                self.trajectory = mdtraj.load(full_paths)
                traj_dcd = os.path.join(data_dir, f'{self.code}.dcd')
                traj_top = os.path.join(data_dir, f'{self.code}.top.pdb')
                self.trajectory.save(traj_dcd)
                self.trajectory[0].save(traj_top)
                print(f"Generated trajectory saved to {data_dir}.")
            else:
                print(f"No PDB files found in directory: {self.data_path}")
        else:
            print(f'Unsupported file format for data file: {self.data_path}')
            return
        # Save a copy of the trajectory for sampling
        self.original_trajectory = self.trajectory
            
    def check_coarse_grained(self):
        topology = self.trajectory.topology
        atoms = topology.atoms
        self.coarse_grained = all(atom.element.symbol == 'C' for atom in atoms)
        self.atom_selector = "protein" if self.coarse_grained else "name == CA"

    def random_sample_trajectory(self, sample_size:int):
        total_frames = len(self.original_trajectory)
        if sample_size > total_frames:
            raise ValueError("Sample size cannot be larger than the total number of frames in the trajectory.")
        random_indices = np.random.choice(total_frames, size=sample_size, replace=False)
        self.trajectory = mdtraj.Trajectory(
            xyz=self.original_trajectory.xyz[random_indices],
            topology=self.original_trajectory.topology)
        print(f"{sample_size} conformations sampled from {self.code} trajectory.")
        
    def extract_features(self, featurization:str, min_sep:int, max_sep:int):
        print(f"Performing feature extraction for Ensemble: {self.code}.")
        self.features, self.names = self.featurize(featurization, min_sep, max_sep, get_names=True)
        print("Transformed ensemble shape:", self.features.shape)

    def featurize(self, featurization: str, min_sep: int, max_sep: int, get_names: bool = False):
        if featurization == "ca_dist":
            return featurize_ca_dist(
                traj=self.trajectory, 
                get_names=get_names,
                atom_selector=self.atom_selector,
                min_sep=min_sep,
                max_sep=max_sep)
        elif featurization == "phi_psi":
            return featurize_phi_psi(
                traj=self.trajectory, 
                get_names=get_names)
        elif featurization == "a_angle":
            return featurize_a_angle(
                traj=self.trajectory, 
                get_names=get_names, 
                atom_selector=self.atom_selector)
        elif featurization == "tr_omega":
            return featurize_tr_angle(
                traj=self.trajectory,
                type="omega",
                get_names=get_names,
                min_sep=min_sep,
                max_sep=max_sep)
        elif featurization == "tr_phi":
            return featurize_tr_angle(
                traj=self.trajectory,
                type="phi",
                get_names=get_names,
                min_sep=min_sep,
                max_sep=max_sep)
        elif featurization == "rg":
            return mdtraj.compute_rg(self.trajectory), None if get_names else mdtraj.compute_rg(self.trajectory)
        else:
            raise NotImplementedError("Unsupported feature extraction method.")
        
    def normalize_features(self, mean, std):
        self.features = (self.features - mean) / std

    def select_chain(self):
        topology = self.trajectory.topology

        if topology.n_chains == 1:
            return

        # Get all unique chain IDs from the topology
        chain_ids = set(chain.index for chain in topology.chains)

        while True:
            if self.chain_id is None:
                print(f"Ensemble {self.code} has multiple chains. Enter the chain ID you want to select. Available chain IDs:", chain_ids)
                sys.stdout.flush()  # Flush the output buffer
                self.chain_id = input("Enter the chain ID you want to select: ")

            try:
                chain_id_input = int(self.chain_id)
                if chain_id_input not in chain_ids:
                    print("Invalid chain ID. Please select from the available options.")
                    self.chain_id = None
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a valid chain ID.")
                self.chain_id = None  # Reset chain_id to None to prompt again for input

        chain_A_indices = topology.select(f"chainid {self.chain_id}")
        self.trajectory = self.trajectory.atom_slice(chain_A_indices)


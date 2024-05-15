import os
import sys
from typing import Sequence, Tuple, Union
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

    residue_range : Tuple, optional
        A tuple indicating the start and end of the residue range (inclusive), using 1-based indexing. Default is None.

    Notes
    -----
    - If the database is 'atlas', the ensemble code should be provided as a PDB ID with a chain identifier separated by an underscore. Example: '3a1g_B'.
    - If the database is 'ped', the ensemble code should be in the PED ID format, which consists of a string starting with 'PED' followed by a numeric identifier, and 'e' followed by another numeric identifier. Example: 'PED00423e001'.
    - 'chain_id' is always assigned in order to the actual chains. For example, chains A, C, D would have 'chain_id' 0, 1, 2 respectively.
    - The `residue_range` parameter uses 1-based indexing, meaning the first residue is indexed as 1.
    """
    def __init__(self, code: str, data_path: str = None, top_path: str = None, database: str = None, chain_id: int = None, residue_range: Tuple = None) -> None:
        self.code = code
        self.data_path = data_path
        self.top_path = top_path
        self.database = database
        self.chain_id = chain_id
        self.residue_range = residue_range
    
    def load_trajectory(self, data_dir: str):  
        """
        Load a trajectory for the ensemble.

        Parameters
        ----------
        data_dir : str
            The directory where the trajectory data is located or where generated trajectory files will be saved.

        Notes
        -----
        This method loads a trajectory for the ensemble based on the specified data path. 
        It supports loading from various file formats such as PDB, DCD, and XTC.
        If the data path points to a directory, it searches for PDB files within the directory 
        and generates a trajectory from them.
        Additional processing steps include checking for coarse-grained models and selecting a single chain if multiple chains were loaded.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data file or directory for ensemble {self.code} doesn't"
                f" exists: {self.data_path}"
            )
        elif self.data_path.endswith('.pdb'):
            print(f"Generating trajectory for {self.code}...")
            self.trajectory = mdtraj.load(self.data_path)
            traj_dcd = os.path.join(data_dir, f'{self.code}.dcd')
            traj_top = os.path.join(data_dir, f'{self.code}.top.pdb')
            self.trajectory.save(traj_dcd)
            self.trajectory[0].save(traj_top)
            print(f"Generated trajectory saved to {data_dir}.")
        elif self.data_path.endswith(('.dcd', '.xtc')) and os.path.exists(self.top_path):
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
        # Save the trajectory for sampling
        self.original_trajectory = self.trajectory
        # Check if a coarse-grained model was loaded
        self._check_coarse_grained()
        # Select a single chain if multiple chains were loaded
        self._select_chain()
        # Select residues
        self._select_residues()
            
    def _check_coarse_grained(self):
        topology = self.trajectory.topology
        atoms = topology.atoms
        self.coarse_grained = all(atom.element.symbol == 'C' for atom in atoms)
        self.atom_selector = "protein" if self.coarse_grained else "name == CA"

    def random_sample_trajectory(self, sample_size: int):
        """
        Randomly sample frames from the original trajectory.

        Parameters
        ----------
        sample_size : int
            The number of frames to sample from the original trajectory.

        Notes
        -----
        This method samples frames randomly from the original trajectory and updates the ensemble's trajectory attribute.
        """
        total_frames = len(self.original_trajectory)
        if sample_size > total_frames:
            raise ValueError("Sample size cannot be larger than the total number of frames in the trajectory.")
        random_indices = np.random.choice(total_frames, size=sample_size, replace=False)
        self.trajectory = mdtraj.Trajectory(
            xyz=self.original_trajectory.xyz[random_indices],
            topology=self.original_trajectory.topology)
        print(f"{sample_size} conformations sampled from {self.code} trajectory.")
        
    def extract_features(self, featurization: str, min_sep: int, max_sep: int):
        """
        Extract features from the trajectory using the specified featurization method.

        Parameters
        ----------
        featurization : str
            The method to use for feature extraction. Supported options: 'ca_dist', 'phi_psi', 'a_angle', 'tr_omega', 'tr_phi', 'rg'.
        min_sep : int
            The minimum sequence separation for angle calculations.
        max_sep : int
            The maximum sequence separation for angle calculations.

        Notes
        -----
        This method extracts features from the trajectory using the specified featurization method and updates the ensemble's features attribute.
        """
        print(f"Performing feature extraction for Ensemble: {self.code}.")
        self.features, self.names = self.get_features(featurization, min_sep, max_sep, get_names=True)
        print("Transformed ensemble shape:", self.features.shape)

    def get_features(self, featurization: str, min_sep: int, max_sep: int, get_names: bool = False) -> Union[Tuple[Sequence, Sequence], Tuple[Sequence, None]]:
        """
        Get features from the trajectory using the specified featurization method.

        Parameters
        ----------
        featurization : str
            The method to use for feature extraction. Supported options: 'ca_dist', 'phi_psi', 'a_angle', 'tr_omega', 'tr_phi', 'rg'.
        min_sep : int
            The minimum sequence separation for angle calculations.
        max_sep : int
            The maximum sequence separation for angle calculations.
        get_names : bool, optional
            Whether to return feature names along with features. Default is False.

        Returns
        -------
        features : Sequence
            The extracted features.
        names : Sequence or None
            If `get_names` is True, returns a sequence of feature names corresponding to the extracted features. Otherwise, returns None.

        Notes
        -----
        This method extracts features from the trajectory using the specified featurization method.
        """
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
        
    def normalize_features(self, mean: float, std: float):
        """
        Normalize the extracted features using the provided mean and standard deviation.

        Parameters
        ----------
        mean : float
            The mean value used for normalization.
        std : float
            The standard deviation used for normalization.

        Notes
        -----
        This method normalizes the ensemble's features using the provided mean and standard deviation.
        """
        self.features = (self.features - mean) / std

    def _select_chain(self):
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

    def _validate_residue_range(self):
        """
        Validate the residue range to ensure it's within the valid range of residues in the trajectory.
        """
        if self.residue_range is None:
            return
        start_residue, end_residue = self.residue_range

        total_residues = self.trajectory.topology.n_residues

        if not (1 <= start_residue <= total_residues):
            raise ValueError(f"Start residue {start_residue} is out of range. Must be between 1 and {total_residues}.")
        if not (1 <= end_residue <= total_residues):
            raise ValueError(f"End residue {end_residue} is out of range. Must be between 1 and {total_residues}.")
        if start_residue > end_residue:
            raise ValueError(f"Start residue {start_residue} must be less than or equal to end residue {end_residue}.")

    def _select_residues(self):
        """
        Modify self.trajectory to only include residues within self.residue_range.
        """
        if self.residue_range is None:
            return
        self._validate_residue_range()
        start_residue, end_residue = self.residue_range
        atom_indices = self.trajectory.topology.select(f'residue >= {start_residue} and residue <= {end_residue}')
        self.trajectory = self.trajectory.atom_slice(atom_indices)
        print(f"Selected residues from ensemble {self.code}")

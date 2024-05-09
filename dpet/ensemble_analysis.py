from pathlib import Path
import re
from typing import Dict, List
import zipfile
from dpet.data.api_client import APIClient
from dpet.ensemble import Ensemble
from dpet.data.extract_tar_gz import extract_tar_gz
import os
import mdtraj
import numpy as np
from dpet.dimensionality_reduction.dimensionality_reduction import DimensionalityReductionFactory

class EnsembleAnalysis:
    """
    Data analysis pipeline for ensemble data.

    Initializes with a list of ensemble objects and a directory path
    for storing data.

    Parameters
    ----------
    ensembles (list[Ensemble]): List of ensembles.
    output_dir (str): Directory path for storing data.
    """
    def __init__(self, ensembles:list[Ensemble], output_dir:str):
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_client = APIClient()
        self.feature_names = []
        self.all_labels = []
        self.ensembles: List[Ensemble] = ensembles

    @property
    def ens_codes(self) -> List[str]:
        """
        Get the ensemble codes.

        Returns
        -------
        List[str]
            A list of ensemble codes.
        """
        return [ensemble.code for ensemble in self.ensembles]

    @property
    def trajectories(self) -> Dict[str, mdtraj.Trajectory]:
        """
        Get the trajectories associated with each ensemble.

        Returns
        -------
        Dict[str, mdtraj.Trajectory]
            A dictionary where keys are ensemble IDs and values are the corresponding MDTraj trajectories.
        """
        return {ensemble.code: ensemble.trajectory for ensemble in self.ensembles}

    @property
    def features(self) -> Dict[str, np.ndarray]:
        """
        Get the features associated with each ensemble.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary where keys are ensemble IDs and values are the corresponding feature arrays.
        """
        return {ensemble.code: ensemble.features for ensemble in self.ensembles}
    
    @property
    def reduce_dim_data(self) -> Dict[str, np.ndarray]:
        """
        Get the transformed data associated with each ensemble.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary where keys are ensemble IDs and values are the corresponding feature arrays.
        """
        return {ensemble.code: ensemble.reduce_dim_data for ensemble in self.ensembles}

    def __del__(self):
        if hasattr(self, 'api_client'):
            self.api_client.close_session()
    
    def _download_from_ped(self, ensemble: Ensemble):
        ped_pattern = r'^(PED\d{5})(e\d{3})$'

        code = ensemble.code
        match = re.match(ped_pattern, code)
        if not match:
            print(f"Entry {code} does not match the PED ID pattern and will be skipped.")
            return
        
        ped_id = match.group(1)
        ensemble_id = match.group(2)
        tar_gz_filename = f'{code}.tar.gz'
        tar_gz_file = os.path.join(self.output_dir, tar_gz_filename)

        pdb_filename = f'{code}.pdb'
        pdb_file = os.path.join(self.output_dir, pdb_filename)

        if not os.path.exists(tar_gz_file) and not os.path.exists(pdb_file):
            url = f'https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb'
            print(f"Downloading entry {code} from PED.")
            headers = {'accept': '*/*'}

            response = self.api_client.perform_get_request(url, headers=headers)
            if response:
                # Download and save the response content to a file
                self.api_client.download_response_content(response, tar_gz_file)
                print(f"Downloaded file {tar_gz_filename} from PED.")
        else:
            print(f"Ensemble {code} already downloaded. Skipping.")

        # Extract the .tar.gz file
        if not os.path.exists(pdb_file):
            extract_tar_gz(tar_gz_file, self.output_dir, pdb_filename)
            print(f"Extracted file {pdb_filename}.")
        else:
            print(f"File {pdb_filename} already exists. Skipping extraction.")
        
        # Set the data path to the downloaded file
        # If the trajectory is already generated it will be used instead of the pdb file
        traj_dcd = os.path.join(self.output_dir, f'{code}.dcd')
        traj_top = os.path.join(self.output_dir, f'{code}.top.pdb')
        if os.path.exists(traj_dcd) and os.path.exists(traj_top):
            print(f'Trajectory already exists for ensemble {code}.')
            ensemble.data_path = traj_dcd
            ensemble.top_path = traj_top
        else:
            ensemble.data_path = pdb_file
    
    def _download_from_atlas(self, ensemble: Ensemble):
        pdb_pattern = r'^\d\w{3}_[A-Z]$'
        code = ensemble.code
        if not re.match(pdb_pattern, code):
            print(f"Entry {code} does not match the PDB ID pattern and will be skipped.")
            return []

        zip_filename = f'{code}.zip'
        zip_file = os.path.join(self.output_dir, zip_filename)

        if not os.path.exists(zip_file):
            print(f"Downloading entry {code} from Atlas.")
            url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{code}/{code}_protein.zip"
            headers = {'accept': '*/*'}

            response = self.api_client.perform_get_request(url, headers=headers)
            if not response:
                return
            # Download and save the response content to a file
            self.api_client.download_response_content(response, zip_file)
            print(f"Downloaded file {zip_filename} from Atlas.")
        else:
            print("File already exists. Skipping download.")

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Map reps to original ensemble code
            zip_contents = zip_ref.namelist()
            new_ensembles = []
            for fname in zip_contents:
                if fname.endswith('.xtc'):
                    new_code = fname.split('.')[0]
                    data_path = os.path.join(self.output_dir, fname)
                    top_path = os.path.join(self.output_dir, f"{code}.pdb")
                    ensemble = Ensemble(code=new_code, data_path=data_path, top_path=top_path)
                    new_ensembles.append(ensemble)
            # Unzip
            zip_ref.extractall(self.output_dir)
            print(f"Extracted file {zip_file}.")

            # Remove unused files.
            for unused_path in self.output_dir.glob("*.tpr"):
                os.remove(unused_path)
            os.remove(os.path.join(self.output_dir, "README.txt"))

        return new_ensembles

    def load_trajectories(self) -> Dict[str, mdtraj.Trajectory]:
        """
        Load trajectories for all ensembles.

        This method iterates over each ensemble in the `ensembles` list and downloads
        data files if they are not already available. 
        Trajectories are then loaded for each ensemble.

        Returns
        -------
        Dict[str, mdtraj.Trajectory]
            A dictionary where keys are ensemble IDs and values are the corresponding MDTraj trajectories.

        Note
        ----
        This method assumes that the `output_dir` attribute of the class specifies the directory
        where trajectory files will be saved or extracted.
        """
        new_ensembles_mapping = {}
        for ensemble in self.ensembles:
            if ensemble.database == 'ped':
                self._download_from_ped(ensemble)
            elif ensemble.database == 'atlas':
                new_ensembles = self._download_from_atlas(ensemble)
                new_ensembles_mapping[ensemble.code] = new_ensembles

        # Update self.ensembles using the mapping
        updated_ensembles = []
        for ensemble in self.ensembles:
            new_ensembles = new_ensembles_mapping.get(ensemble.code, [ensemble])
            updated_ensembles.extend(new_ensembles)
        self.ensembles = updated_ensembles
        
        for ensemble in self.ensembles:
            ensemble.load_trajectory(self.output_dir)
        
        return self.trajectories

    def random_sample_trajectories(self, sample_size: int):
        """
        Sample a defined random number of conformations from the ensemble 
        trajectory. 

        Parameters
        ----------
        sample_size: int
            Number of conformations sampled from the ensemble. 
        """
        for ensemble in self.ensembles:
            ensemble.random_sample_trajectory(sample_size)
        return self.trajectories

    def extract_features(self, featurization: str, normalize: bool = False, min_sep: int = 2, max_sep: int = None) -> Dict[str, np.ndarray]:
        """
        Extract the selected feature.

        Parameters
        ----------
        featurization : str
            Choose between "phi_psi", "ca_dist", "a_angle", "tr_omega", and "tr_phi".

        normalize : bool, optional
            Whether to normalize the data. Only applicable to the "ca_dist" method. Default is False.

        min_sep : int, optional
            Minimum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is 2.

        max_sep : int, optional
            Maximum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is None.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary where keys are ensemble IDs and values are the corresponding feature arrays.
        """
        self._featurize(featurization=featurization, min_sep=min_sep, max_sep=max_sep)
        self._create_all_labels()
        if normalize and featurization == "ca_dist":
            self._normalize_data()
        return self.features

    def exists_coarse_grained(self) -> bool:
        """
        Check if at least one of the loaded ensembles is coarse-grained after loading trajectories.

        Returns
        -------
        bool
            True if at least one ensemble is coarse-grained, False otherwise.
        """
        return any(ensemble.coarse_grained for ensemble in self.ensembles)

    def _featurize(self, featurization: str, min_sep, max_sep):
        if featurization in ("phi_psi", "tr_omega", "tr_phi") and self.exists_coarse_grained():
            raise ValueError(f"{featurization} feature extraction is not possible when working with coarse-grained models.")
        self.featurization = featurization
        for ensemble in self.ensembles:
            ensemble.extract_features(featurization, min_sep, max_sep)
        self.feature_names = list(self.ensembles)[0].names
        print("Feature names:", self.feature_names)

    def _create_all_labels(self):
        self.all_labels = []
        for ensemble in self.ensembles:
            num_data_points = len(ensemble.features)
            self.all_labels.extend([ensemble.code] * num_data_points)

    def _normalize_data(self):
        mean = self.concat_features.mean(axis=0)
        std = self.concat_features.std(axis=0)
        self.concat_features = (self.concat_features - mean) / std
        for ensemble in self.ensembles:
            ensemble.normalize_features(mean, std)

    def _calculate_rg_for_trajectory(self, trajectory:mdtraj.Trajectory):
        return [mdtraj.compute_rg(frame) for frame in trajectory]

    @property
    def rg(self) -> list[float]:
        """
        Calculates Rg for each conformation in the loaded ensembles.
        The returned values are in Angstrom.  

        Returns
        -------
        list[float]
            A list of Rg values for each conformation in the loaded ensembles, in Angstrom.
        """
        rg_values_list = []
        for ensemble in self.ensembles:
            traj = ensemble.trajectory
            rg_values_list.extend(self._calculate_rg_for_trajectory(traj))
        return [item[0] * 10 for item in rg_values_list]

    def _get_concat_features(self, fit_on: list[str]=None):
        if fit_on and any(f not in self.ens_codes for f in fit_on):
            raise ValueError("Cannot fit on ensembles that were not provided as input.")
        if fit_on is None:
            fit_on = self.ens_codes
        concat_features = [ensemble.features for ensemble in self.ensembles if ensemble.code in fit_on]
        concat_features = np.concatenate(concat_features, axis=0)
        print("Concatenated featurized ensemble shape:", concat_features.shape)
        return concat_features

    def reduce_features(self, method: str, fit_on:list[str]=None, *args, **kwargs) -> np.ndarray:
        """
        Perform dimensionality reduction on the extracted features.

        Parameters
        ----------
        method : str
            Choose between "pca", "tsne", "dimenfix", "mds", and "kpca".

        fit_on : list[str], optional
            if method is "pca" or "kpca", specifies on which ensembles the models should be fit. 
            The model will then be used to transform all ensembles.

        Additional Parameters
        ---------------------
        The following optional parameters apply based on the selected reduction method:

        - pca:
            - num_dim : int, optional
                Number of components to keep. Default is 10.

        - tsne:
            - perplexity_vals : list[float], optional
                List of perplexity values. Default is range(2, 10, 2).
            - metric : str, optional
                Metric to use. Default is "euclidean". 
            - circular : bool, optional
                Whether to use circular metrics. Default is False.
            - n_components : int, optional
                Number of dimensions of the embedded space. Default is 2.
            - learning_rate : float, optional
                Learning rate. Default is 100.0.
            - range_n_clusters : list[int], optional
                Range of cluster values. Default is range(2, 10, 1).

        - dimenfix:
            - range_n_clusters : list[int], optional
                Range of cluster values. Default is range(1, 10, 1).

        - mds:
            - num_dim : int, optional
                Number of dimensions. Default is 2.

        - kpca:
            - circular : bool, optional
                Whether to use circular metrics. Default is False.
            - num_dim : int, optional
                Number of components to keep. Default is 10.
            - gamma : float, optional
                Kernel coefficient. Default is None.

        Returns
        -------
        np.ndarray
            Returns the transformed data.

        For more information on each method, see the corresponding documentation:
            - PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            - t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
            - DimenFix: https://github.com/visml/neo_force_scheme/tree/0.0.3
            - MDS: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
            - Kernel PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
        """
        # Check if all ensemble features have the same size
        feature_sizes = set(ensemble.features.shape[1] for ensemble in self.ensembles)
        if len(feature_sizes) > 1:
            print("Error: Features from ensembles have different sizes. Cannot concatenate.")
            return None
        self.concat_features = self._get_concat_features()
        self.reducer = DimensionalityReductionFactory.get_reducer(method, *args, **kwargs)
        self.reduce_dim_method = method
        if method in ("pca","kpca"):
            fit_on_data = self._get_concat_features(fit_on=fit_on)
            self.reduce_dim_model = self.reducer.fit(data=fit_on_data)
            for ensemble in self.ensembles:
                ensemble.reduce_dim_data = self.reducer.transform(ensemble.features)
                print("Reduced dimensionality ensemble shape:", ensemble.reduce_dim_data.shape)
            self.transformed_data = self.reducer.transform(data=self.concat_features)
        else:
            self.transformed_data = self.reducer.fit_transform(data=self.concat_features)
        return self.transformed_data

    def execute_pipeline(self, featurization_params:dict, reduce_dim_params:dict, subsample_size:int=None):
        """
        Execute the data analysis pipeline end-to-end. The pipeline includes:
            1. Download from database (optional)
            2. Generate trajectories
            3. Randomly sample a number of conformations from trajectories (optional)
            4. Perform feature extraction
            5. Perform dimensionality reduction

        Parameters
        ----------
        featurization_params: dict
            Parameters for feature extraction. The only required parameter is "featurization",
            which can be "phi_psi", "ca_dist", "a_angle", "tr_omega" or "tr_phi". 
            Other method-specific parameters are optional.
        reduce_dim_params: dict
            Parameters for dimensionality reduction. The only required parameter is "method",
            which can be "pca", "tsne", "dimenfix", "mds" or "kpca".
        subsample_size: int, optional
            Optional parameter that specifies the trajectory subsample size. Default is None.
        """
        self.load_trajectories()
        if subsample_size is not None:
            self.random_sample_trajectories(subsample_size)
        self.extract_features(**featurization_params)
        self.reduce_features(**reduce_dim_params)

    def get_features(self, featurization: str, min_sep: int = 2, max_sep: int = None) -> Dict[str, np.ndarray]:
        """
        Extract features for each ensemble without modifying any fields in the EnsembleAnalysis class.

        Parameters:
        -----------
        featurization : str
            The type of featurization to be applied. Supported options are "phi_psi", "tr_omega", "tr_phi", "ca_dist", "a_angle" and "rg".

        min_sep : int, optional
            Minimum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is 2.

        max_sep : int, optional
            Maximum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is None.

        Returns:
        --------
        Dict[str, np.ndarray]
            A dictionary containing the extracted features for each ensemble, where the keys are ensemble IDs and the 
            values are NumPy arrays containing the features.
        """
        if featurization in ("phi_psi", "tr_omega", "tr_phi") and self.exists_coarse_grained():
            raise ValueError(f"{featurization} feature extraction is not possible when working with coarse-grained models.")
        
        features_dict = {}
        for ensemble in self.ensembles:
            features = ensemble.get_features(featurization=featurization, min_sep= min_sep, max_sep=max_sep)
            features_dict[ensemble.code] = features
        return features_dict
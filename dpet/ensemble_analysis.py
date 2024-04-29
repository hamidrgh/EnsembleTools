from pathlib import Path
import re
import shutil
from typing import Dict
import zipfile
from dpet.data.api_client import APIClient
from dpet.ensemble import Ensemble
from dpet.data.extract_tar_gz import extract_tar_gz
import os
import mdtraj
import numpy as np
from dpet.dimensionality_reduction.dimensionality_reduction import DimensionalityReductionFactory

class EnsembleAnalysis:
    def __init__(self, ens_codes:list[str], data_dir:str):
        self.data_dir = Path(data_dir)
        self.api_client = APIClient()
        self.feature_names = []
        self.all_labels = []
        self.ens_codes = ens_codes
        self.ensembles: Dict[str, Ensemble] = {}

    @property
    def trajectories(self) -> Dict[str, mdtraj.Trajectory]:
        return {ens_id: ensemble.trajectory for ens_id, ensemble in self.ensembles.items()}

    @property
    def features(self) -> Dict[str, np.array]:
        return {ens_id: ensemble.features for ens_id, ensemble in self.ensembles.items()}

    def __del__(self):
        if hasattr(self, 'api_client'):
            self.api_client.close_session()
    
    def download_from_ped(self):
        """
        Automate Downloading ensembles using PED API 

        Note
        ----
        The function only downloads ensembles in the PED ID format, which consists of a string starting with 'PED'
        followed by a numeric identifier and 'e' followed by another numeric identifier.
        Example: 'PED00423e001', 'PED00424e001'
        """
        ped_pattern = r'^(PED\d{5})(e\d{3})$'

        # Filter the ens_codes list using regex
        for ens_code in self.ens_codes:
            match = re.match(ped_pattern, ens_code)
            if not match:
                print(f"Entry {ens_code} does not match the PED ID pattern and will be skipped.")
                continue
            
            ped_id = match.group(1)
            ensemble_id = match.group(2)
            tar_gz_filename = f'{ens_code}.tar.gz'
            tar_gz_file = os.path.join(self.data_dir, tar_gz_filename)

            pdb_filename = f'{ens_code}.pdb'
            pdb_file = os.path.join(self.data_dir, pdb_filename)

            if not os.path.exists(tar_gz_file) and not os.path.exists(pdb_file):
                url = f'https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb'
                print(f"Downloading entry {ens_code} from PED.")
                headers = {'accept': '*/*'}

                response = self.api_client.perform_get_request(url, headers=headers)
                if response:
                    # Download and save the response content to a file
                    self.api_client.download_response_content(response, tar_gz_file)
                    print(f"Downloaded file {tar_gz_filename} from PED.")
            else:
                print(f"Ensemble {ens_code} already downloaded. Skipping.")

            # Extract the .tar.gz file
            if not os.path.exists(pdb_file):
                extract_tar_gz(tar_gz_file, self.data_dir, pdb_filename)
                print(f"Extracted file {pdb_filename}.")
            else:
                print(f"File {pdb_filename} already exists. Skipping extraction.")
    
    def download_from_atlas(self):
        """ 
        Automate Downloading MD ensembles from Atlas. 

        Note
        ----
        The function only downloads ensembles provided as PDB ID with a chain identifier separated by an underscore.
        Example: '3a1g_B'
        """
        pdb_pattern = r'^\d\w{3}_[A-Z]$'
        new_ens_codes_mapping = {}
        for ens_code in self.ens_codes:

            if not re.match(pdb_pattern, ens_code):
                print(f"Entry {ens_code} does not match the PDB ID pattern and will be skipped.")
                continue

            zip_filename = f'{ens_code}.zip'
            zip_file = os.path.join(self.data_dir, zip_filename)

            if not os.path.exists(zip_file):
                print(f"Downloading entry {ens_code} from Atlas.")
                url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{ens_code}/{ens_code}_protein.zip"
                headers = {'accept': '*/*'}

                response = self.api_client.perform_get_request(url, headers=headers)
                if not response:
                    continue
                # Download and save the response content to a file
                self.api_client.download_response_content(response, zip_file)
                print(f"Downloaded file {zip_filename} from Atlas.")
            else:
                print("File already exists. Skipping download.")

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Map reps to original ensemble code
                zip_contents = zip_ref.namelist()
                new_ens_codes = [fname.split('.')[0] for fname in zip_contents if fname.endswith('.xtc')]
                new_ens_codes_mapping[ens_code] = new_ens_codes
                # Unzip
                zip_ref.extractall(self.data_dir)
                print(f"Extracted file {zip_file}.")

            # Remove unused files.
            for unused_path in self.data_dir.glob("*.tpr"):
                os.remove(unused_path)
            os.remove(os.path.join(self.data_dir, "README.txt"))

            # Copy and rename topology file
            old_pdb_file = os.path.join(self.data_dir,f"{ens_code}.pdb")
            for new_code in new_ens_codes:
                new_pdb_file = os.path.join(self.data_dir,f"{new_code}.top.pdb")
                shutil.copy(old_pdb_file, new_pdb_file)
                print(f"Copied and renamed {old_pdb_file} to {new_pdb_file}.")
            # Delete old topology file
            os.remove(old_pdb_file)

        # Update self.ens_codes using the mapping
        updated_ens_codes = []
        for old_code in self.ens_codes:
            updated_ens_codes.extend(new_ens_codes_mapping.get(old_code, [old_code]))

        self.ens_codes = updated_ens_codes
        print("Analysing ensembles:", self.ens_codes)

    def download_from_database(self, database: str =None):
        """ 
        Download ensembles from databases.

        Parameters
        ----------
        database : str
            Choose the database you want to download from ('ped'/'atlas').

        Note
        ----
        For PED database:
            The function only downloads ensembles in the PED ID format, which consists of a string starting with 'PED'
            followed by a numeric identifier and 'e' followed by another numeric identifier.
            Example: 'PED00423e001', 'PED00424e001'

        For atlas database:
            The function only downloads ensembles provided as PDB ID with a chain identifier separated by an underscore.
            Example: '3a1g_B'
        """
        if database == "ped":
            self.download_from_ped()
        elif database == "atlas":
            self.download_from_atlas()

    def generate_trajectories(self):
        """
        Load trajectory files into mdtraj objects.
        
        Supported file formats:
            1. [ens_code].dcd (trajectory file) + [ens_code].top.pdb (topology file)
            2. [ens_code].xtc (trajectory file) + [ens_code].top.pdb (topology file)
            3. [ens_code].pdb
            4. Directory [ens_code] containing several .pdb files
        
        For each ensemble code (ens_code):
            - If both trajectory (.dcd or .xtc) and topology (.top.pdb) files exist, load the trajectory.
            - If only a .pdb file exists, generate trajectory and topology files from the .pdb file.
            - If a directory [ens_code] exists containing .pdb files, generate trajectory and topology files from the directory.

        Note
        ----
        Using 'download_from_database' transforms the downloaded data into the appropriate format.
        """
        for ens_code in self.ens_codes:
            ensemble = Ensemble(ens_code, self.data_dir)
            ensemble.load_trajectory()
            ensemble.select_chain()
            ensemble.check_coarse_grained()
            self.ensembles[ens_code] = ensemble
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
        for ensemble in self.ensembles.values():
            ensemble.random_sample_trajectory(sample_size)
        return self.trajectories

    def extract_features(self, featurization: str, normalize: bool = False, min_sep: int = 2, max_sep: int = None):
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
        """
        self._featurize(featurization=featurization, min_sep=min_sep, max_sep=max_sep)
        self.concat_features = self._get_concat_features()
        self._create_all_labels()
        if normalize and featurization == "ca_dist":
            self._normalize_data()
        return self.features

    def exists_coarse_grained(self):
        """
        Check if at least one of the loaded ensembles is coarse-grained after loading trajectories.
        """
        return any(ensemble.coarse_grained for ensemble in self.ensembles.values())

    def _featurize(self, featurization: str, min_sep, max_sep):
        if featurization in ("phi_psi", "tr_omega", "tr_phi") and self.exists_coarse_grained():
            raise ValueError(f"{featurization} feature extraction is not possible when working with coarse-grained models.")
        self.featurization = featurization
        for ensemble in self.ensembles.values():
            ensemble.extract_features(featurization, min_sep, max_sep)
        self.feature_names = list(self.ensembles.values())[0].names
        print("Feature names:", self.feature_names)

    def _create_all_labels(self):
        self.all_labels = []
        for ens_id, ensemble in self.ensembles.items():
            num_data_points = len(ensemble.features)
            self.all_labels.extend([ens_id] * num_data_points)

    def _normalize_data(self):
        mean = self.concat_features.mean(axis=0)
        std = self.concat_features.std(axis=0)
        self.concat_features = (self.concat_features - mean) / std
        for ensemble in self.ensembles.values():
            ensemble.normalize_features(mean, std)

    def _calculate_rg_for_trajectory(self, trajectory:mdtraj.Trajectory):
        return [mdtraj.compute_rg(frame) for frame in trajectory]

    @property
    def rg(self):
        """
        Calculates Rg for each conformations in the loaded ensembles.
        The returned values are in Angstrom.  
        """
        rg_values_list = []
        for ensemble in self.ensembles.values():
            traj = ensemble.trajectory
            rg_values_list.extend(self._calculate_rg_for_trajectory(traj))
        return [item[0] * 10 for item in rg_values_list]

    def _get_concat_features(self, fit_on: list[str]=None):
        if fit_on and any(f not in self.ens_codes for f in fit_on):
            raise ValueError("Cannot fit on ensembles that were not provided as input.")
        if fit_on is None:
            fit_on = self.ens_codes
        concat_features = [ensemble.features for ens_code, ensemble in self.ensembles.items() if ens_code in fit_on]
        concat_features = np.concatenate(concat_features, axis=0)
        print("Concatenated featurized ensemble shape:", concat_features.shape)
        return concat_features

    def reduce_features(self, method: str, fit_on:list[str]=None, *args, **kwargs):
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

        For more information on each method, see the corresponding documentation:
            - PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            - t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
            - DimenFix: https://github.com/visml/neo_force_scheme/tree/0.0.3
            - MDS: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
            - Kernel PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
        """

        self.reducer = DimensionalityReductionFactory.get_reducer(method, *args, **kwargs)
        self.reduce_dim_method = method
        if method in ("pca","kpca"):
            fit_on_data = self._get_concat_features(fit_on=fit_on)
            self.reduce_dim_model = self.reducer.fit(data=fit_on_data)
            self.reduce_dim_data = {}
            for ensemble in self.ensembles.values():
                ensemble.reduce_dim_data = self.reducer.transform(ensemble.features)
                print("Reduced dimensionality ensemble shape:", ensemble.reduce_dim_data.shape)
            self.transformed_data = self.reducer.transform(data=self.concat_features)
        else:
            self.transformed_data = self.reducer.fit_transform(data=self.concat_features)
        return self.transformed_data

    def execute_pipeline(self, featurization_params:dict, reduce_dim_params:dict, database:str=None, subsample_size:int=None):
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
        database: str, optional
            Optional parameter that specifies the database from which the entries should be downloaded.
            Options are "ped" and "atlas". Default is None.
        subsample_size: int, optional
            Optional parameter that specifies the trajectory subsample size. Default is None.
        """
        self.download_from_database(database)
        self.generate_trajectories()
        if subsample_size is not None:
            self.random_sample_trajectories(subsample_size)
        self.extract_features(**featurization_params)
        self.reduce_features(**reduce_dim_params)
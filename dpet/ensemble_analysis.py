from pathlib import Path
import re
from typing import Dict, List, Optional, Union, Tuple
import zipfile
from dpet.featurization.distances import rmsd
import pandas as pd
from dpet.data.api_client import APIClient
from dpet.ensemble import Ensemble
from dpet.data.extract_tar_gz import extract_tar_gz
import os
import mdtraj
import numpy as np
from dpet.dimensionality_reduction import DimensionalityReductionFactory
from dpet.featurization.ensemble_level import ensemble_features
import itertools
from dpet.data.comparison import (
    score_avg_jsd, score_emd_approximation, get_num_comparison_bins
)

class EnsembleAnalysis:
    """
    Data analysis pipeline for ensemble data.

    Initializes with a list of ensemble objects and a directory path
    for storing data.

    Parameters
    ----------
    ensembles : List[Ensemble])
        List of ensembles.
    output_dir : str
        Directory path for storing data.
    """
    def __init__(self, ensembles:List[Ensemble], output_dir:str):
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_client = APIClient()
        self.feature_names = []
        self.all_labels = []
        self.ensembles: List[Ensemble] = ensembles
        self.param_feat = None

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
    
    def __getitem__(self, code):
        for e in self.ensembles:
            if e.code == code:
                return e
        raise KeyError(f"Ensemble with code '{code}' not found")

    def __del__(self):
        if hasattr(self, 'api_client'):
            self.api_client.close_session()

    def _download_from_ped(self, ensemble: Ensemble):
        ped_pattern = r'^(PED\d{5})(e\d{3})$'

        code = ensemble.code
        match = re.match(ped_pattern, code)
        if not match:
            raise ValueError(f"Entry {code} does not match the PED ID pattern.")
        
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
            if response is None:
                raise ConnectionError(f"Failed to connect to PED server for entry {code}.")
            if response.status_code != 200:
                raise ConnectionError(f"Failed to download entry {code} from PED. HTTP status code: {response.status_code}")
            
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
        ensemble.data_path = pdb_file      
        chain_ids = ensemble.get_chains_from_pdb()
        if len(chain_ids) > 1 and ensemble.chain_id is not None and ensemble.chain_id in chain_ids:
            traj_suffix = f'_{ensemble.chain_id.upper()}'
        else:
            traj_suffix = ''

        traj_dcd = os.path.join(self.output_dir, f'{ensemble.code}{traj_suffix}.dcd')
        traj_top = os.path.join(self.output_dir, f'{ensemble.code}{traj_suffix}.top.pdb')

        if os.path.exists(traj_dcd) and os.path.exists(traj_top):
            print(f'Trajectory file already exists for ensemble {code}.')
            ensemble.data_path = traj_dcd
            ensemble.top_path = traj_top

    def _download_from_atlas(self, ensemble: Ensemble):
        pdb_pattern = r'^\d\w{3}_[A-Z]$'
        code = ensemble.code
        if not re.match(pdb_pattern, code):
            raise ValueError(f"Entry {code} does not match the PDB ID pattern.")

        zip_filename = f'{code}.zip'
        zip_file = os.path.join(self.output_dir, zip_filename)

        if not os.path.exists(zip_file):
            print(f"Downloading entry {code} from Atlas.")
            url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{code}/{code}_protein.zip"
            headers = {'accept': '*/*'}

            response = self.api_client.perform_get_request(url, headers=headers)
            if response is None:
                raise ConnectionError(f"Failed to connect to Atlas server for entry {code}.")
            if response.status_code != 200:
                raise ConnectionError(f"Failed to download entry {code} from Atlas. HTTP status code: {response.status_code}")
            
            # Download and save the response content to a file
            self.api_client.download_response_content(response, zip_file)
            print(f"Downloaded file {zip_filename} from Atlas.")
        else:
            print("File already exists. Skipping download.")

        try:
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
                readme_path = os.path.join(self.output_dir, "README.txt")
                if os.path.exists(readme_path):
                    os.remove(readme_path)

        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(f"Failed to unzip file {zip_file}. The file may be corrupted.")

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
            elif ensemble.database is None:
                pass
            else:
                raise KeyError(f"Unknown database: {ensemble.database}")

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

    def _join_ensemble_traj(self, atom_selector = 'backbone'):
        merge_traj = []
        for traj in self.trajectories:
            
            atom_indices = self.trajectories[traj].topology.select(atom_selector)
            new_ca_traj = self.trajectories[traj].atom_slice(atom_indices)
            merge_traj.append(new_ca_traj)
        joined_traj = mdtraj.join(merge_traj, check_topology=False, discard_overlapping_frames=False)
        
        return joined_traj


    def extract_features(self, featurization: str, normalize: bool = False, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract the selected feature.

        Parameters
        ----------
        featurization : str
            Choose between "phi_psi", "ca_dist", "a_angle", "tr_omega", "tr_phi", "rmsd".

        normalize : bool, optional
            Whether to normalize the data. Only applicable to the "ca_dist" method. Default is False.

        min_sep : int or None, optional
            Minimum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is 2.

        max_sep : int, optional
            Maximum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is None.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary where keys are ensemble IDs and values are the corresponding feature arrays.
        """
        if featurization == 'rmsd':
            self.param_feat = 'rmsd'
            j_traj = self._join_ensemble_traj()
            rmsd_matrix = rmsd(j_traj)
            self._create_all_labels()
            return rmsd_matrix
            
        else:
            self.param_feat = featurization
            self._featurize(featurization=featurization, *args, **kwargs)
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

    def _featurize(self, featurization: str, *args, **kwargs):
        if featurization in ("phi_psi", "tr_omega", "tr_phi") and self.exists_coarse_grained():
            raise ValueError(f"{featurization} feature extraction is not possible when working with coarse-grained models.")
        self.featurization = featurization
        for ensemble in self.ensembles:
            ensemble.extract_features(featurization, *args, **kwargs)
        self.feature_names = list(self.ensembles)[0].names
        print("Feature names:", self.feature_names)




    def _create_all_labels(self):
        self.all_labels = []
        for ensemble, traj in zip(self.ensembles,self.trajectories):
            num_data_points = self.trajectories[traj].n_frames
            self.all_labels.extend([ensemble.code] * num_data_points)

    def _normalize_data(self):
        feature_sizes = set(ensemble.features.shape[1] for ensemble in self.ensembles)
        if len(feature_sizes) > 1:
            raise ValueError("Error: Features from ensembles have different sizes. Cannot normalize data.")
        self.concat_features = self._get_concat_features()
        mean = self.concat_features.mean(axis=0)
        std = self.concat_features.std(axis=0)
        self.concat_features = (self.concat_features - mean) / std
        for ensemble in self.ensembles:
            ensemble.normalize_features(mean, std)

    def _get_concat_features(self, fit_on: List[str]=None):
        if fit_on and any(f not in self.ens_codes for f in fit_on):
            raise ValueError("Cannot fit on ensembles that were not provided as input.")
        if fit_on is None:
            fit_on = self.ens_codes
        concat_features = [ensemble.features for ensemble in self.ensembles if ensemble.code in fit_on]
        concat_features = np.concatenate(concat_features, axis=0)
        print("Concatenated featurized ensemble shape:", concat_features.shape)
        return concat_features

    def reduce_features(self, method: str, fit_on:List[str]=None, *args, **kwargs) -> np.ndarray:
        """
        Perform dimensionality reduction on the extracted features.

        Parameters
        ----------
        method : str
            Choose between "pca", "tsne", "dimenfix", "mds", "kpca" and "umap".

        fit_on : List[str], optional
            if method is "pca" or "kpca", specifies on which ensembles the models should be fit. 
            The model will then be used to transform all ensembles.

        Additional Parameters
        ---------------------
        The following optional parameters apply based on the selected reduction method:

        - pca:
            - num_dim : int, optional
                Number of components to keep. Default is 10.

        - tsne:
            - perplexity_vals : List[float], optional
                List of perplexity values. Default is range(2, 10, 2).
            - metric : str, optional
                Metric to use. Default is "euclidean". 
            - circular : bool, optional
                Whether to use circular metrics. Default is False.
            - n_components : int, optional
                Number of dimensions of the embedded space. Default is 2.
            - learning_rate : float, optional
                Learning rate. Default is 100.0.
            - range_n_clusters : List[int], optional
                Range of cluster values. Default is range(2, 10, 1).

        - dimenfix:
            - range_n_clusters : List[int], optional
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
            - UMAP: https://umap-learn.readthedocs.io/en/latest/
        """

        if self.param_feat == 'rmsd':
            self.reducer = DimensionalityReductionFactory.get_reducer(method, *args, **kwargs)
            self.reduce_dim_method = method
            self.transformed_data = self.reducer.fit_transform(data=self.extract_features(featurization=self.param_feat))
            return self.transformed_data
        # Check if all ensemble features have the same size
        # Check if all ensemble features have the same size
        else:
            
            feature_sizes = set(ensemble.features.shape[1] for ensemble in self.ensembles)
            if len(feature_sizes) > 1:
                raise ValueError("Features from ensembles have different sizes. Cannot concatenate.")

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

    def execute_pipeline(self, featurization_params:Dict, reduce_dim_params:Dict, subsample_size:int=None):
        """
        Execute the data analysis pipeline end-to-end. The pipeline includes:
            1. Download from database (optional)
            2. Generate trajectories
            3. Randomly sample a number of conformations from trajectories (optional)
            4. Perform feature extraction
            5. Perform dimensionality reduction

        Parameters
        ----------
        featurization_params: Dict
            Parameters for feature extraction. The only required parameter is "featurization",
            which can be "phi_psi", "ca_dist", "a_angle", "tr_omega" or "tr_phi". 
            Other method-specific parameters are optional.
        reduce_dim_params: Dict
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

    def get_features(self, featurization: str, normalize: bool = False, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract features for each ensemble without modifying any fields in the EnsembleAnalysis class.

        Parameters:
        -----------
        featurization : str
            The type of featurization to be applied. Supported options are "phi_psi", "tr_omega", "tr_phi", "ca_dist", "a_angle", "rg", "prolateness", "asphericity", "sasa", "end_to_end" and "flory_exponent".

        min_sep : int, optional
            Minimum sequence separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is 2.

        max_sep : int or None, optional
            Maximum sequence separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is None.

        normalize : bool, optional
            Whether to normalize the extracted features. Normalization is only supported when featurization is "ca_dist". Default is False.

        Returns:
        --------
        Dict[str, np.ndarray]
            A dictionary containing the extracted features for each ensemble, where the keys are ensemble IDs and the 
            values are NumPy arrays containing the features.

        Raises:
        -------
        ValueError:
            If featurization is not supported, or if normalization is requested for a featurization method other than "ca_dist".
            If normalization is requested and features from ensembles have different sizes.
            If coarse-grained models are used with featurization methods that require atomistic detail.
        """
        if featurization in ("phi_psi", "tr_omega", "tr_phi") and self.exists_coarse_grained():
            raise ValueError(f"{featurization} feature extraction is not possible when working with coarse-grained models.")
        
        if normalize and featurization not in ("ca_dist", "end_to_end"):
            raise ValueError("Normalization is only supported when featurization is 'ca_dist'.")
        
        features_dict = {}
        for ensemble in self.ensembles:
            features = ensemble.get_features(featurization=featurization, normalize=normalize, *args, **kwargs)
            if featurization != "flory_exponent":
                features_dict[ensemble.code] = features
            else:
                features_dict[ensemble.code] = features[0]
            
        if normalize and featurization == "ca_dist":
            feature_sizes = set(features.shape[1] for features in features_dict.values())
            if len(feature_sizes) > 1:
                raise ValueError("Error: Features from ensembles have different sizes. Cannot normalize data.")
            concat_features = np.concatenate(list(features_dict.values()), axis=0)
            mean = concat_features.mean(axis=0)
            std = concat_features.std(axis=0)
            for key, features in features_dict.items():
                features_dict[key] = (features - mean) / std
        
        return features_dict
    
    def get_features_summary_dataframe(self, selected_features: List[str] = ["rg", "asphericity", "prolateness", "sasa", "end_to_end", "flory_exponent"], show_variability: bool = True) -> pd.DataFrame:
        """
        Create a summary DataFrame for each ensemble.

        The DataFrame includes the ensemble code and the average for each feature.

        Parameters
        ----------
        selected_features : List[str], optional
            List of feature extraction methods to be used for summarizing the ensembles.
            Default is ["rg", "asphericity", "prolateness", "sasa", "end_to_end", "flory_exponent"].
        show_variability: bool, optional
            If True, include a column  a measurment of variability for each
            feature (e.g.: standard deviation or error).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the summary statistics (average and std) for each feature in each ensemble.
        
        Raises
        ------
        ValueError
            If any feature in the selected_features is not a supported feature extraction method.
        """
        supported_features = {"rg", "asphericity", "prolateness", "sasa", "end_to_end", "ee_on_rg", "flory_exponent"}

        # Validate the selected_features
        invalid_features = [feature for feature in selected_features if feature not in supported_features]
        if invalid_features:
            raise ValueError(f"Unsupported feature extraction methods: {', '.join(invalid_features)}")

        summary_data = []

        for ensemble in self.ensembles:
            ensemble_code = ensemble.code
            summary_row = [
                ensemble_code,
                ensemble.trajectory.n_residues,
                len(ensemble.trajectory)
            ]
            
            for feature in selected_features:
                features = ensemble.get_features(featurization=feature, normalize=False)
                if feature not in ensemble_features:
                    features_array = np.array(features)
                    feature_mean = features_array.mean()
                    feature_std = features_array.std()
                    summary_row.extend([feature_mean, feature_std])
                else:
                    summary_row.extend([features[0], features[1]])
            
            summary_data.append(summary_row)

        columns = ['ensemble_code', 'n_residues', 'n_conformers']
        for feature in selected_features:
            if feature not in ensemble_features:
                columns.extend([f"{feature}_mean", f"{feature}_std"])
            else:
                columns.extend([feature, f"{feature}_err"])

        summary_df = pd.DataFrame(summary_data, columns=columns)
        if not show_variability:
            summary_df = summary_df[[c for c in summary_df.columns \
                                     if not c.endswith(("_std", "_err"))]]
        
        return summary_df
    
    
    def comparison_scores(
            self,
            score: str,
            feature: str,
            featurization_params: dict = {},
            bootstrap_iters: int = 3,
            bootstrap_frac: float = 1.0,
            bootstrap_replace: bool = True,
            bins: Union[int, str] = 50,
            random_seed: int = None,
            verbose: bool = False
        ) -> Tuple[np.ndarray, List[str]]:
        """
        Compare all pair of ensembles using divergence/distance scores.
        Implemented scores are approximate average Jensen–Shannon divergence
        (JSD) [https://doi.org/10.1371/journal.pcbi.1012144], and approximate
        Earth Mover's Distance (EMD) [https://doi.org/10.1038/s41467-023-36443-x]
        over several kinds of molecular features. The lower these scores are,
        the higher the similarity between the probability distribution of the
        features of the ensembles.

        Parameters
        ----------
        score: str
            Type of score used to compare ensembles. Choices: `jsd` (average
            Jensen–Shannon divergence) and `emd` (Earth Mover's Distance).
        feature: str
            Type of feature to analyze. Choices: `ca_dist` for all sets of Ca-Ca
            distances between non-neighboring residues and `alpha_angle` for 
            all torsion angles between four consecutive Ca atoms in a chain.
        featurization_params: dict, optional
            Optional dictionary to customize the featurization process for the
            above features.
        bootstrap_iters: int, optional
            Number of bootstrap iterations. If the value is > 1, each pair of
            ensembles will be compared `bootstrap_iters` times by randomly
            selecting (bootstrapping) conformations from it. For small protein
            structural ensembles (less than 500-1,000 conformations) comparison
            scores in DPED are not robust estimators of divergence/distance. By
            performing bootstrapping, you can have an idea of how the size of
            your ensembles impacts the comparison. Specifically, you should
            look at auto-comparisons: each ensemble will be compared with itself
            by bootstrapping different subsamples. The scores obtained by these
            auto-comparisons give an estimate of the "uncertainty" related to
            sample size: for large ensembles, the auto-comparison scores (on the
            diagonal of the matrix) should tend to 0. For smaller ensembles you
            will get increasingly higher values. Use values >= 5 especially when
            comparing small ensembles. When comparing large ensembles (more than
            10,000 conformations) you can avoid bootstrapping.
        bootstrap_frac: float, optional
            Fraction of the total conformations to sample when bootstrapping.
            Default value is 1.0, which results in bootstrap samples with the
            same number of conformations of the original ensemble.
        bootstrap_replace: bool, optional
            If `True`, bootstrap will sample with replacement. Default is `True`.
        bins: Union[int, str], optional
            Number of bins or bin assignment rule for JSD comparisons. See the
            documentation of `dpet.data.comparison.get_num_comparison_bins` for
            more information.
        random_seed: int, optional
            Random seed used when performing bootstrapping.
        verbose: bool, optional
            If `True`, some information about the comparisons will be printed to
            stdout.

        Returns
        -------
        results: Tuple[np.ndarray, List[str]]
            A tuple containing:
                `score_matrix`: a (M, M, B) NumPy array storing the comparison
                scores, where M is the number of ensembles being compared and B
                is the number of bootstrap iterations (B will be 1 if
                bootstrapping was not performed).
                `codes`: a list of M strings, containing the codes of the
                ensembles that were compared.
        """

        allowed_scores = {
            "jsd": ["ca_dist", "alpha_angle"],
            "emd": ["ca_dist", "alpha_angle"]
        }
        
        ### Check arguments.
        if score == "jsd":
            score_func = score_avg_jsd
        elif score == "emd":
            score_func = score_emd_approximation
        else:
            raise ValueError(
                "The type of similarity score should be selected among:"
                f" {list(allowed_scores.keys())}"
            )
        if not feature in allowed_scores[score]:
            raise ValueError(
                f"The '{score}' score must be calculated based on the"
                f" following features: {allowed_scores[score]}"
            )

        if not 0 <= bootstrap_frac <= 1:
            raise ValueError(f"Invalid bootstrap_frac: {bootstrap_frac}")

        ### Check the ensembles.
        num_residues = set([e.get_num_residues() for e in self.ensembles])
        if len(num_residues) != 1:
            raise ValueError(
                "Can only compare ensembles with the same number of residues."
                " Ensembles in this analysis have different number of residues."
            )
        
        ### Define the random seed.
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
            rand_func = rng.choice
        else:
            rand_func = np.random.choice

        ### Featurize (run it here to avoid re-calculating at every comparison).
        features = []
        # Compute features.
        for ensemble_i in self.ensembles:
            if feature == "ca_dist":
                feats_i = ensemble_i.get_features(
                    normalize=False,
                    featurization="ca_dist",
                    min_sep=featurization_params.get("min_sep", 2),
                    max_sep=featurization_params.get("max_sep"),
                )
            elif feature == "alpha_angle":
                feats_i = ensemble_i.get_features(featurization="a_angle", normalize=False)
            else:
                raise ValueError(f"Invalid feature for comparison: {feature}")
            features.append(feats_i)
        
        ### Performs the comparisons.
        if verbose:
            print(f"# Scoring '{score}' using features '{feature}'")
        codes = list(self.trajectories.keys())
        n = len(codes)

        # Define the parameters for the evaluation.
        if score == "jsd":
            # Apply the same bin number to every comparison, based on the number
            # of conformers in the smallest ensemble.
            num_bins = get_num_comparison_bins(bins, x=features)
            scoring_params = {"bins": num_bins}
            if verbose:
                print(f"- Number of bins for all comparisons: {num_bins}")
        elif score == "emd":
            if feature in ("alpha_angle", ):  # Angular features.
                scoring_params = {"metric": "angular_l2"}
            else:  # Euclidean features.
                scoring_params = {"metric": "rmsd"}
            if verbose:
                print(f"- Distance function for comparing: {scoring_params['metric']}")
        else:
            raise ValueError(score)

        # Initialize a (n, n, bootstrap_iters) matrix.
        comparisons_iters = 1 if bootstrap_iters < 2 else bootstrap_iters
        score_matrix = np.zeros((n, n, comparisons_iters))

        # Get the pairs to compare.
        pairs_to_compare = []
        for i, code_i in enumerate(codes):
            for j, code_j in enumerate(codes):
                if j >= i:
                    if j != i or bootstrap_iters > 1:
                        pairs_to_compare.append((i, j))
        if verbose:
            print(
                f"- Will perform: {len(pairs_to_compare)} (pairs of ensembles)"
                f" x {comparisons_iters} (iterations) ="
                f" {len(pairs_to_compare)*comparisons_iters} (comparisons)"
            )

        # Compare pairs of ensembles.
        for i, j in pairs_to_compare:
            for k in range(comparisons_iters):
                # Use all conformers.
                if bootstrap_iters == 0:
                    features_ik = features[i]
                    features_jk = features[j]
                # When using bootstrapping, subsample the ensembles.
                else:
                    # Features for ensemble i.
                    n_i = features[i].shape[0]
                    rand_ids_ik = rand_func(
                        n_i,
                        max(int(n_i*bootstrap_frac), 1),
                        replace=bootstrap_replace
                    )
                    features_ik = features[i][rand_ids_ik]
                    # Features for ensemble j.
                    n_j = features[j].shape[0]
                    rand_ids_jk = rand_func(
                        n_j,
                        max(int(n_j*bootstrap_frac), 1),
                        replace=bootstrap_replace
                    )
                    features_jk = features[j][rand_ids_jk]
                # Score.
                score_ijk = score_func(
                    features_ik, features_jk, **scoring_params
                )
                # Store the results.
                score_matrix[i, j, k] = score_ijk
                score_matrix[j, i, k] = score_ijk

        return score_matrix, codes
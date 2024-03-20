from api_client import APIClient
from visualization import dimenfix_cluster_scatter_plot, dimenfix_cluster_scatter_plot_2, dimenfix_scatter_plot, dimenfix_scatter_plot_2, pca_correlation_plot, pca_cumulative_explained_variance, pca_plot_1d_histograms, pca_plot_2d_landscapes, pca_rg_correlation, plot_average_dmap_comparison, plot_cmap_comparison, plot_distance_distribution_multiple, trajectories_plot_asphericity, trajectories_plot_density, trajectories_plot_dihedrals, trajectories_plot_prolateness, trajectories_plot_relative_helix_content_multiple_proteins, trajectories_plot_rg_comparison, trajectories_plot_total_sasa, trajectories_scatter_prolateness, tsne_ramachandran_plot, tsne_ramachandran_plot_density, tsne_scatter_plot, tsne_scatter_plot_2
from utils import extract_tar_gz
from ped_entry import PedEntry
import os
import mdtraj
from featurizer import FeaturizationFactory
import numpy as np
from dimensionality_reduction import DimensionalityReductionFactory

DIM_REDUCTION_DIR = "dim_reduction"
PDB_DIR = "pdb_data"
TRAJ_DIR = "traj"

class EnsembleAnalysis:
    def __init__(self, ped_entries: PedEntry, data_dir: str):
        self.data_dir = data_dir
        self.api_client = APIClient()
        self.trajectories = {}
        self.feature_names = []
        self.featurized_data = {}
        self.all_labels = []
        self.generate_ens_codes(ped_entries)
        self.ped_entries = ped_entries

    def __del__(self):
        if hasattr(self, 'api_client'):
            self.api_client.close_session()
    
    def generate_ens_codes(self, ped_entries):
        self.ens_codes = []
        for ped_entry in ped_entries:
            ped_id = ped_entry.ped_id
            ensemble_ids = ped_entry.ensemble_ids
            for ensemble_id in ensemble_ids:
                self.ens_codes.append(f"{ped_id}{ensemble_id}")

    def download_from_ped(self):
        for ped_entry in self.ped_entries:
            ped_id = ped_entry.ped_id
            for ensemble_id in ped_entry.ensemble_ids:
                ens_code = f"{ped_id}{ensemble_id}"
                tar_gz_filename = f'{ens_code}.tar.gz'
                tar_gz_file = os.path.join(self.data_dir, tar_gz_filename)

                pdb_dir = os.path.join(self.data_dir, PDB_DIR)
                pdb_filename = f'{ens_code}.pdb'
                pdb_file = os.path.join(pdb_dir, pdb_filename)

                if not os.path.exists(tar_gz_file) and not os.path.exists(pdb_file):
                    url = f'https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb'
                    headers = {'accept': '*/*'}

                    response = self.api_client.perform_get_request(url, headers=headers)
                    if response:
                        # Download and save the response content to a file
                        self.api_client.download_response_content(response, tar_gz_file)
                        print(f"Downloaded file {tar_gz_filename} from PED.")
                else:
                    print("File already exists. Skipping download.")

                # Extract the .tar.gz file
                if not os.path.exists(pdb_file):
                    extract_tar_gz(tar_gz_file, pdb_dir, pdb_filename)
                    print(f"Extracted file {pdb_filename}.")
                else:
                    print("File already exists. Skipping extracting.")

    def generate_trajectories(self):
        pdb_dir = os.path.join(self.data_dir, PDB_DIR)
        traj_dir = os.path.join(self.data_dir, TRAJ_DIR)
        os.makedirs(traj_dir, exist_ok=True)
    
        for ens_code in self.ens_codes:    
            pdb_filename = f'{ens_code}.pdb'
            pdb_file = os.path.join(pdb_dir, pdb_filename)

            traj_dcd = os.path.join(traj_dir, f'{ens_code}.dcd')
            traj_top = os.path.join(traj_dir, f'{ens_code}.top.pdb')

            # Generate trajectory from pdb if it doesn't exist, otherwise load it.
            if not os.path.exists(traj_dcd) and not os.path.exists(traj_top):
                print(f'Generating trajectory from {pdb_filename}.')
                trajectory = mdtraj.load(pdb_file)
                print(f'Saving trajectory.')
                trajectory.save(traj_dcd)
                trajectory[0].save(traj_top)
            else:
                print(f'Trajectory already exists. Loading trajectory.')
                trajectory = mdtraj.load(traj_dcd, top=traj_top)

            self.trajectories[ens_code] = trajectory

    def perform_feature_extraction(self, featurization: str, normalize = False, *args, **kwargs):
        self.extract_features(featurization, *args, **kwargs)
        self.concatenate_features()
        self.create_all_labels()
        if normalize and featurization == "ca_dist":
            self.normalize_data()

    def extract_features(self, featurization: str, *args, **kwargs):
        featurizer = FeaturizationFactory.get_featurizer(featurization, *args, **kwargs)
        get_names = True
        self.featurization = featurization
        for ens_code, trajectory in self.trajectories.items():
            print(f"Performing feature extraction for Ensemble: {ens_code}.")
            if get_names:
                features, names = featurizer.featurize(trajectory, get_names=get_names)
                get_names = False
            else:
                features = featurizer.featurize(trajectory, get_names=get_names)
            self.featurized_data[ens_code] = features
            print("Transformed ensemble shape:", features.shape)
        self.feature_names = names
        print("Feature names:", names)

    def concatenate_features(self):
        concat_features = [features for ens_id, features in self.featurized_data.items()]
        self.concat_features = np.concatenate(concat_features, axis=0)
        print("Concatenated featurized ensemble shape:", self.concat_features.shape)
        
    def create_all_labels(self):
        for label, data_points in self.featurized_data.items():
            num_data_points = len(data_points)
            self.all_labels.extend([label] * num_data_points)

    def normalize_data(self):
        mean = self.concat_features.mean(axis=0)
        std = self.concat_features.std(axis=0)
        self.concat_features = (self.concat_features - mean) / std
        for label, features in self.featurized_data.items():
            self.featurized_data[label] = (features - mean) / std

    def calculate_rg_for_trajectory(self, trajectory):
        return [mdtraj.compute_rg(frame) for frame in trajectory]

    def rg_calculator(self):
        rg_values_list = []
        for traj in self.trajectories.values():
            rg_values_list.extend(self.calculate_rg_for_trajectory(traj))
            self.rg = [item[0] * 10 for item in rg_values_list]
        return self.rg

    def fit_dimensionality_reduction(self, method: str, *args, **kwargs):
        dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
        self.reducer = DimensionalityReductionFactory.get_reducer(method, dim_reduction_dir, *args, **kwargs)
        self.reduce_dim_method = method
        if method == "pca":
            self.reduce_dim_model = self.reducer.fit(data=self.concat_features)
            self.reduce_dim_data = {}
            for key, data in self.featurized_data.items():
                self.reduce_dim_data[key] = self.reducer.transform(data)
                print("Reduced dimensionality ensemble shape:", self.reduce_dim_data[key].shape)
            self.transformed_data = self.reducer.transform(data=self.concat_features)
        else:
            self.transformed_data = self.reducer.fit_transform(data=self.concat_features)

    def cluster(self, range_n_clusters):
        self.sil_scores = self.reducer.cluster(range_n_clusters=range_n_clusters)

    def execute_pipeline(self, featurization_params, reduce_dim_params, clustering_params=None):
        self.download_from_ped()
        self.generate_trajectories()
        self.perform_feature_extraction(**featurization_params)
        self.rg_calculator()
        self.fit_dimensionality_reduction(**reduce_dim_params)
        if clustering_params:
            self.cluster(**clustering_params)

    ##################### Integrated plot function #####################

    def tsne_ramachandran_plot(self):
        if self.reduce_dim_method == "tsne":
            dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
            tsne_ramachandran_plot(dim_reduction_dir, self.concat_features)
        else:
            print("Analysis is only valid for t-SNE dimensionality reduction.")

    def tsne_ramachandran_plot_density(self):
        dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
        tsne_ramachandran_plot_density(dim_reduction_dir, self.concat_features)

    def tsne_scatter_plot(self):
        if self.reduce_dim_method == "tsne":
            dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
            tsne_scatter_plot(dim_reduction_dir, self.all_labels, self.ens_codes, self.rg)
        else:
            print("Analysis is only valid for t-SNE dimensionality reduction.")

    def tsne_scatter_plot_2(self):
        dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
        tsne_scatter_plot_2(dim_reduction_dir, self.rg)

    def dimenfix_scatter_plot(self):
        dimenfix_scatter_plot(self.transformed_data, self.rg)

    def dimenfix_scatter_plot_2(self):
        dimenfix_scatter_plot_2(self.transformed_data, self.all_labels)

    def dimenfix_cluster_scatter_plot(self):
        dimenfix_cluster_scatter_plot(self.sil_scores, self.transformed_data)

    def dimenfix_cluster_scatter_plot_2(self):
        dimenfix_cluster_scatter_plot_2(self.sil_scores, self.transformed_data, self.ens_codes, self.all_labels)

    def pca_cumulative_explained_variance(self):
        if self.reduce_dim_method == "pca":
            pca_cumulative_explained_variance(self.reduce_dim_model)
        else:
            print("Analysis is only valid for PCA dimensionality reduction.")

    def pca_plot_2d_landscapes(self):
        dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
        pca_plot_2d_landscapes(self.ens_codes, self.reduce_dim_data, dim_reduction_dir, self.featurization)

    def pca_plot_1d_histograms(self):
        dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
        pca_plot_1d_histograms(self.ens_codes, self.transformed_data, self.reduce_dim_data, dim_reduction_dir, self.featurization)

    def pca_correlation_plot(self, num_residues, sel_dims):
        if self.featurization == "ca_dist":
            pca_correlation_plot(num_residues, sel_dims, self.feature_names, self.reduce_dim_model)
        else:
            print("Analysis is only valid for ca_dist feature extraction.")
    
    def pca_rg_correlation(self):
        dim_reduction_dir = os.path.join(self.data_dir, DIM_REDUCTION_DIR)
        pca_rg_correlation(self.ens_codes, self.trajectories, self.reduce_dim_data, dim_reduction_dir)
        
    def trajectories_plot_total_sasa(self):
        trajectories_plot_total_sasa(self.trajectories)

    def trajectories_plot_asphericity(self):
        trajectories_plot_asphericity(self.trajectories)

    def trajectories_plot_density(self):
        trajectories_plot_density(self.trajectories)

    def trajectories_scatter_prolateness(self):
        trajectories_scatter_prolateness(self.trajectories)

    def trajectories_plot_prolateness(self):
        trajectories_plot_prolateness(self.trajectories)
    
    def trajectories_plot_dihedrals(self):
        trajectories_plot_dihedrals(self.trajectories)

    def trajectories_plot_relative_helix_content_multiple_proteins(self):
        trajectories_plot_relative_helix_content_multiple_proteins(self.trajectories)

    def trajectories_plot_rg_comparison(self, n_bins=50, bins_range=(1, 4.5), dpi=96):
        trajectories_plot_rg_comparison(self.trajectories, n_bins, bins_range, dpi)

    def plot_average_dmap_comparison(self, 
                                    ticks_fontsize=14,
                                    cbar_fontsize=14,
                                    title_fontsize=14,
                                    dpi=96,
                                    max_d=6.8,
                                    use_ylabel=True):
        plot_average_dmap_comparison(self.trajectories, ticks_fontsize, cbar_fontsize, title_fontsize, dpi, max_d, use_ylabel)

    def plot_cmap_comparison(self,
                            title,
                            ticks_fontsize=14,
                            cbar_fontsize=14,
                            title_fontsize=14,
                            dpi=96,
                            cmap_min=-3.5,
                            use_ylabel=True):
        plot_cmap_comparison(self.trajectories, title, ticks_fontsize, cbar_fontsize, title_fontsize, dpi, cmap_min, use_ylabel)

    def plot_distance_distribution_multiple(self, dpi = 96):
        plot_distance_distribution_multiple(self.trajectories, dpi)
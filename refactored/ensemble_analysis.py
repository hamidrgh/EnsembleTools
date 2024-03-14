from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from api_client import APIClient
from visualization import dimenfix_scatter_plot, tsne_ramachandran_plot, tsne_ramachandran_plot_density, tsne_scatter_plot, tsne_scatter_plot_2
from utils import extract_tar_gz
from ped_entry import PedEntry
import os
import mdtraj
from featurizer import FeaturizationFactory
import numpy as np
from dimensionality_reduction import DimensionalityReductionFactory

class EnsembleAnalysis:
    def __init__(self, ped_entries: PedEntry, data_dir: str):
        self.ped_entries = ped_entries
        self.data_dir = data_dir
        self.api_client = APIClient()
        self.trajectories = {}
        self.feature_names = []
        self.featurized_data = {}
        self.all_labels = []

    def __del__(self):
        if hasattr(self, 'api_client'):
            self.api_client.close_session()

    def download_from_ped(self):
        for ped_entry in self.ped_entries:
            ped_id = ped_entry.ped_id
            ensemble_ids = ped_entry.ensemble_ids
            for ensemble_id in ensemble_ids:

                generated_name = f'{ped_id}_{ensemble_id}'
                tar_gz_filename = f'{generated_name}.tar.gz'
                tar_gz_file = os.path.join(self.data_dir, tar_gz_filename)

                pdb_dir = os.path.join(self.data_dir, 'pdb_data')
                pdb_filename = f'{generated_name}.pdb'
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
        pdb_dir = os.path.join(self.data_dir, 'pdb_data')
        traj_dir = os.path.join(self.data_dir, 'traj')
        os.makedirs(traj_dir, exist_ok=True)
    
        for ped_entry in self.ped_entries:
            ped_id = ped_entry.ped_id
            ensemble_ids = ped_entry.ensemble_ids
            for ensemble_id in ensemble_ids:
                
                generated_name = f'{ped_id}_{ensemble_id}'
                pdb_filename = f'{generated_name}.pdb'
                pdb_file = os.path.join(pdb_dir, pdb_filename)

                traj_dcd = os.path.join(traj_dir, f'{generated_name}.dcd')
                traj_top = os.path.join(traj_dir, f'{generated_name}.top.pdb')

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

                self.trajectories[(ped_id, ensemble_id)] = trajectory

    def perform_feature_extraction(self, featurization: str, normalize = False, *args, **kwargs):
        self.extract_features(featurization, *args, **kwargs)
        self.concatenate_features()
        self.create_all_labels()
        if normalize and featurization == "ca_dist":
            self.normalize_data()

    def extract_features(self, featurization: str, *args, **kwargs):
        featurizer = FeaturizationFactory.get_featurizer(featurization, *args, **kwargs)
        get_names = True
        for (ped_id, ensemble_id), trajectory in self.trajectories.items():
            print(f"Performing feature extraction for PED ID: {ped_id}, ensemble ID: {ensemble_id}.")
            if get_names:
                features, names = featurizer.featurize(trajectory, get_names=get_names)
                get_names = False
            else:
                features = featurizer.featurize(trajectory, get_names=get_names)
            self.featurized_data[(ped_id, ensemble_id)] = features
            print("Transformed ensemble shape:", features.shape)
        self.feature_names = names
        print("Feature names:", names)

    def concatenate_features(self):
        concat_features = [features for (_, _), features in self.featurized_data.items()]
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
        reducer = DimensionalityReductionFactory.get_reducer(method, *args, **kwargs)
        if method == "pca":
            self.reduce_dim_model = reducer.fit(data=self.concat_features)
            self.reduce_dim_data = {}
            for key, data in self.featurized_data.items():
                self.reduce_dim_data[key] = reducer.transform(data)
                print("Reduced dimensionality ensemble shape:", self.reduce_dim_data[key].shape)
            self.concat_reduce_dim_data = reducer.transform(data=self.concat_features)
        else:
            self.transformed_data = reducer.fit_transform(data=self.concat_features)

    def create_tsne_clusters(self, perplexityVals, range_n_clusters, tsne_dir):
        # TODO: Remove tsne_dir from arguments
        for perp in perplexityVals:
            tsne = np.loadtxt(tsne_dir + '/tsnep'+str(perp))
            for n_clusters in range_n_clusters:
                # print("n_clusters",n_clusters)
                kmeans = KMeans(n_clusters=n_clusters, n_init= 'auto').fit(tsne)
                np.savetxt(tsne_dir + '/kmeans_'+str(n_clusters)+'clusters_centers_tsnep'+str(perp), kmeans.cluster_centers_, fmt='%1.3f')
                np.savetxt(tsne_dir + '/kmeans_'+str(n_clusters)+'clusters_tsnep'+str(perp)+'.dat', kmeans.labels_, fmt='%1.1d')
                
                # print("Kmeans",kmeans,kmeans.labels_)
                #### Compute silhouette score based on low-dim and high-dim distances        
                silhouette_ld = silhouette_score(tsne, kmeans.labels_)
                silhouette_hd = silhouette_score(self.concat_features, kmeans.labels_)
                # print(silhouette_ld)
                with open(tsne_dir + '/silhouette.txt', 'a') as f:
                    f.write("\n")
                    print(perp, n_clusters, silhouette_ld, silhouette_hd, silhouette_ld*silhouette_hd, file =f)

    def tsne_ramachandran_plot(self, tsne_dir):
        tsne_ramachandran_plot(tsne_dir, self.concat_features)

    def tsne_ramachandran_plot_density(self, tsne_dir):
        tsne_ramachandran_plot_density(tsne_dir, self.concat_features)

    def tsne_scatter_plot(self, tsne_dir):
        tsne_scatter_plot(tsne_dir, self.all_labels, self.featurized_data.keys(), self.rg)

    def tsne_scatter_plot_2(self, tsne_dir):
        tsne_scatter_plot_2(tsne_dir, self.rg)

    def dimenfix_scatter_plot(self):
        dimenfix_scatter_plot(self.transformed_data, self.rg)

    def execute_pipeline(self, featurization_params, reduce_dim_params, clustering_params=None):
        self.download_from_ped()
        self.generate_trajectories()
        self.perform_feature_extraction(**featurization_params)
        self.rg_calculator()
        self.fit_dimensionality_reduction(**reduce_dim_params)
        if reduce_dim_params.get('method') == 'tsne':
            self.create_tsne_clusters(**clustering_params)
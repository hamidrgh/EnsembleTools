from api_client import APIClient
from utils import extract_tar_gz
from ped_entry import PedEntry
import os
import mdtraj
from featurizer import FeaturizationFactory
import numpy as np

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

    def perform_feature_extraction(self, featurization: str, seq_sep:int=2, inverse:bool=False, normalize:bool=False):
        self.extract_features(featurization, seq_sep, inverse)
        self.concatenate_features()
        self.create_all_labels()
        if normalize and featurization == "ca_dist":
            self.normalize_data()

    def extract_features(self, featurization: str, seq_sep:int=2, inverse:bool=False):
        featurizer = FeaturizationFactory.get_featurizer(featurization, seq_sep=seq_sep, inverse=inverse)
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
        return [item[0] * 10 for item in rg_values_list]

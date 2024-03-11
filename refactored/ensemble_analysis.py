from api_client import APIClient
from utils import extract_tar_gz, read_file
from ped_entry import PedEntry
import os
import mdtraj
from featurizer import FeaturizationFactory

class EnsembleAnalysis:
    def __init__(self, ped_entries: PedEntry, data_dir: str):
        self.ped_entries = ped_entries
        self.data_dir = data_dir
        self.api_client = APIClient()
        self.trajectories = {}

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

    def perform_feature_extraction(self, featurization: str):
        featurizer = FeaturizationFactory.get_featurizer(featurization)
        for (ped_id, ensemble_id), trajectory in self.trajectories.items():
            ret = featurizer.featurize(trajectory, get_names=True)
            print(ret)


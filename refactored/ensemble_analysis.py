from api_client import APIClient
from utils import extract_tar_gz, read_file
from ped_entry import PedEntry
import os

class EnsembleAnalysis:
    def __init__(self, ped_entries:PedEntry, data_dir:str):
        self.ped_entries = ped_entries
        self.data_dir = data_dir
        self.api_client = APIClient()

    def run_analysis(self):
        for ped_entry in self.ped_entries:
            ped_id = ped_entry.ped_id
            ensemble_ids = ped_entry.ensemble_ids
            for ensemble_id in ensemble_ids:
                url = f'https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb'
                headers = {'accept': '*/*'}

                response = self.api_client.perform_get_request(url, headers=headers)
                if response:
                    content_length = response.headers.get('content-length')
                    content_type = response.headers.get('content-type')
                    print("Content-Length:", content_length)
                    print("Content-Type:", content_type)

                    generated_name = f'{ped_id}_{ensemble_id}'
                    filename = f'{generated_name}.tar.gz'
                    tar_gz_file = os.path.join(self.data_dir, filename)

                    # Download and save the response content to a file
                    self.api_client.download_response_content(response, tar_gz_file)
                    self.api_client.close_session()

                    output_dir = os.path.join(self.data_dir, 'pdb_data')
                    pdb_filename = f'{generated_name}.pdb'
                    # Extract the .tar.gz file
                    extract_tar_gz(tar_gz_file, output_dir, pdb_filename)

                    pdb_file = os.path.join(output_dir, pdb_filename)
                    if os.path.exists(pdb_file):
                        print("Reading contents of", pdb_filename)
                        read_file(pdb_file)
                    else:
                        print("File not found:", pdb_file)


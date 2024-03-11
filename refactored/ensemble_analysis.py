from api_client import APIClient
import os
from utils import extract_tar_gz, read_pdb_file
import gzip

class EnsembleAnalysis:
    def __init__(self, ped_id, ensemble_id, data_dir) -> None:
        api_client = APIClient()
        url = f'https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb'
        headers = {'accept': '*/*'}

        response = api_client.perform_get_request(url, headers=headers)
        if response:
            content_length = response.headers.get('content-length')
            content_type = response.headers.get('content-type')
            print("Content-Length:", content_length)
            print("Content-Type:", content_type)

            # Specify the path to save the downloaded file
            generated_name = f'{ped_id}_{ensemble_id}'
            filename = f'{generated_name}.tar.gz'
            tar_gz_file = os.path.join(data_dir, filename)

            # Download and save the response content to a file
            api_client.download_response_content(response, tar_gz_file)

            output_dir = os.path.join(data_dir, generated_name)
            pdb_filename = 'pdbfile.pdb'

            # Extract the .tar.gz file
            extract_tar_gz(tar_gz_file, output_dir)

            # Access the entire directory or read a single .pdb file
            if os.path.isdir(os.path.join(output_dir, pdb_filename)):
                # Access the entire directory
                print("Entire directory contents:")
                for entry in os.listdir(os.path.join(output_dir, pdb_filename)):
                    print(entry)
            else:
                # Read a single .pdb file
                pdb_file = os.path.join(output_dir, pdb_filename)
                if os.path.exists(pdb_file):
                    print("Content of", pdb_filename)
                    print(read_pdb_file(pdb_file))
                else:
                    print("File not found:", pdb_file)
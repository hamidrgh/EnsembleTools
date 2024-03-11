import tarfile
import os

def extract_tar_gz(tar_gz_file, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the .tar.gz file
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        tar.extractall(output_dir)

def read_pdb_file(pdb_file):
    # Read the content of the .pdb file
    with open(pdb_file, 'r') as f:
        content = f.read()
    return content

# Example usage:
tar_gz_file = '/path/to/your/directory.tar.gz'
output_dir = '/path/to/output/directory'
pdb_filename = 'example.pdb'


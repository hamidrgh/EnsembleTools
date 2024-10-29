import os
import tarfile

def extract_tar_gz(tar_gz_file:str, output_dir:str, new_name:str):
    # Extract the .pdb file with renaming
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        for member in tar.getmembers():
            if os.path.splitext(member.name)[1] == '.pdb':
                member.name = new_name
                tar.extract(member, path=output_dir)
                break  # Only rename and extract the first .pdb file

import os
import tarfile

def extract_tar_gz(tar_gz_file, output_dir, new_name=None):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the .pdb file
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        pdb_member = next((member for member in tar.getmembers() if os.path.splitext(member.name)[1] == '.pdb'), None)
        if pdb_member:
            # Set the extracted file name
            extracted_name = new_name if new_name else 'pdbfile.pdb'
            extracted_path = os.path.join(output_dir, extracted_name)
            
            # Extract the .pdb file
            tar.extract(pdb_member, path=output_dir)
            # Rename the extracted file
            os.rename(os.path.join(output_dir, pdb_member.name), extracted_path)
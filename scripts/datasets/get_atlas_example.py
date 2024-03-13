"""
For development only. I was not able to download PED ensembles via Python, so
i downloaded some ensembles from the ATLAS database:

https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/3a1g_B/3a1g_B.html

Those are 3 MD ensembles of a small protein with little disorder. This is only
for testing right now, not to be included in the final version.

Usage:
Go in the root directory of the repository and run:
    python scripts/datasets/get_atlas_example.py
"""

import os
import pathlib
import subprocess
import mdtraj


# Setup.
data_path = pathlib.Path("ensemble_files")
if not data_path.is_dir():
    raise FileNotFoundError(
        f"Data directory not found. Make a directory called '{data_path.name}'"
        " in the root of the repository.")
prot_path = data_path / "3a1g_B"
if not prot_path.is_dir():
    os.mkdir(prot_path)

# Download.
zip_path = pathlib.Path("3a1g_B_protein.zip")
if zip_path.is_file():
    os.remove(zip_path)
download_cmd = [
    "wget", "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/3a1g_B/3a1g_B_protein.zip"]
subprocess.run(download_cmd)

# Unzip.
unzip_cmd = ["unzip", "-o", str(zip_path), "-d", str(prot_path)]
subprocess.run(unzip_cmd)

# Subsample the ensembles (original ones are too large).
for traj_path in prot_path.glob("./*.xtc"):
    print(f"processing trajectory: {traj_path.name}")
    # Load the full trajectory with 10000 frames (too many).
    traj = mdtraj.load(str(traj_path), top=str(prot_path / "3a1g_B.pdb"))[:-1]
    print("original len:", len(traj))
    # Subsample to 200 frames.
    traj = traj[::50]
    print("after subsampling len:", len(traj))
    traj.save(str(traj_path))

# Remove unused files.
for unused_path in prot_path.glob("./*.tpr"):
    os.remove(unused_path)
os.remove(prot_path / "README.txt")
os.remove(zip_path)

print("done")
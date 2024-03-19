import os
import mdtraj


def read_pdb(input_fp: str) -> mdtraj.Trajectory:

    raise NotImplementedError()


def read_pdb_dir(input_dp: str,
                 code: str) -> mdtraj.Trajectory:
    
    ens_dcd_i = os.path.join(input_dp, f"{code}.dcd")
    ens_top_i = os.path.join(input_dp, f"{code}.top.pdb")

    if not os.path.isfile(ens_dcd_i):
        ens_fp_i = os.path.join(input_dp, f"{code}.pdb")
        print(f"# Loading {ens_fp_i}.")
        trajectory_ens_fp_i =  mdtraj.load(ens_fp_i)
        print("- Saving a DCD file for faster reading next time.")
        trajectory_ens_fp_i.save(ens_dcd_i)
        trajectory_ens_fp_i[0].save(ens_top_i)
        return trajectory_ens_fp_i
    else:
        print(f"# Loading {ens_dcd_i}.")
        trajectory = mdtraj.load(ens_dcd_i, top=ens_top_i)
        return trajectory



traj_formats = ("dcd", "xtc", )

def read_traj(
        input_dp: str,
        code: str,
        topology_fp: str,
        ext: str = None) -> mdtraj.Trajectory:
    """
    `input_dp`: directory where all the trajectory files for an ensembles are.
    `code`: basename of the ensemble file.
    `topology_fp`: topology file oath.
    `ext`: extension of the input files. If not provided, it will be derived
        automatically.
    """
    # Attempts to find trajectory files when the extension if not defined.
    if ext is None:
        ens_file = None
        for traj_format_i in traj_formats:  # Scans for every file extension.
            _ens_file = os.path.join(input_dp, f"{code}.{traj_format_i}")
            if os.path.isfile(_ens_file):
                ens_file = _ens_file
                break
        if ens_file is None:
            raise FileNotFoundError(
                f"No trajectory file (with formats: {', '.join(traj_formats)})"
                f" was found in '{input_dp}' for '{code}'")
    # Uses a user-defined extention to look for trajectory files.
    else:
        ens_file = os.path.join(input_dp, f"{code}.{ext}")
        if not os.path.isfile(ens_file):
            raise FileNotFoundError(
                f"No trajectory file (with format: {ext}) was found in"
                f" '{input_dp}' for '{code}'"
            )
    return mdtraj.load(ens_file, top=topology_fp)
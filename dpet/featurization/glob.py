import numpy as np
import mdtraj


def rg_calculator(traj: mdtraj.Trajectory) -> np.ndarray:
    return mdtraj.compute_rg(traj)
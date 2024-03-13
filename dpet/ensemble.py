from typing import Union, Tuple, List, Callable
import numpy as np
import mdtraj

from dpet.featurization.distances import featurize_ca_dist
from dpet.featurization.angles import featurize_phi_psi
from dpet.featurization.glob import rg_calculator

class Ensemble:
    """
    Class to represent a structural ensemble of a protein.
    """


    def __init__(
            self,
            traj: mdtraj.Trajectory,
            code: str):
        """
        `traj`: mdtraj trajectory object.
        `code`: code.
        """
        self.traj = traj
        self.code = code


    def __len__(self) -> int:
        """
        When using the 'len' function, returns the number of conformers.
        """
        return len(self.traj)


    def rand_subsample(self, num: int) -> None:
        """
        Select a random subset of conformations and remove the others.
        """
        rand_ids = np.random.choice(len(self.traj), num)
        self.traj = mdtraj.Trajectory(
            xyz=self.traj.xyz[rand_ids],
            topology=self.traj.topology)


    def featurize(self,
            featurization: Union[str, Callable],
            get_names: bool = True,
            ravel: bool = True,
            *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Featurize a whole ensemble, whose xyz coordinates are stored in
        the 'traj' object.

        `featurization`: name of the featurization scheme. It must be a scheme
            built-in in the dpet package. Alternatively, users can provide a
            callable (for a user-defined featurization function).
        `get_names`: return also a list of names for each feature.
        `ravel`: ravels the features. If True, the method will return an array
            of dimension (N, num_feats), where all features are raveled into a
            vector of num_feats dimensions. Takes effect only for 'phi_psi'.

        returns: either a numpy array storing a series of features for each
            conformation, or tuple containing a numpy array with features and
            a list of names for the features.
        """

        #--------------------------------------
        # Multiple features for conformation. -
        #--------------------------------------

        if featurization == "ca_dist":
            return featurize_ca_dist(
                traj=self.traj,
                get_names=get_names,
                *args, **kwargs)

        elif featurization == "phi_psi":
            return featurize_phi_psi(
                traj=self.traj,
                get_names=get_names,
                ravel=ravel,
                *args, **kwargs)

        elif featurization == "a_angle":
            # return featurize_a_angle(traj, get_names=get_names, *args, **kwargs)
            raise NotImplementedError()

        #-------------------------------------------------
        # Global features, one feature per conformation. -                      
        #-------------------------------------------------

        elif featurization == "rg":

            return rg_calculator(self.traj)

        #--------------------------------------
        # 
        #--------------------------------------

        elif getattr(featurization, "__call__"):
            raise NotImplementedError(
                "User-defined featurization function, not implemented yet the way it's handled")

        else:
            raise KeyError(
                f"Invalid featurization scheme: {featurization}")
    

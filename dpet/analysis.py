"""
Module with a class implementing a customizable pipeline for readily analyzing
structural ensembles.
"""
import os
from typing import Union, List, Callable, Tuple
from collections import OrderedDict
import numpy as np
from dpet.data.reader import read_traj, read_pdb, read_pdb_dir
from dpet.ensemble import Ensemble
from dpet.reduce_dim import DimensionalityReduction
# The logger module is used mainly used for compatibility with old code
# (to emulate streamlit). Should we keep it?
# My opinion is that it's probably good for development (printing out stuff),
# but users won't need it...?
from dpet.logger import stream as st


####################
# TODO.            #
####################

class FeaturizationOutput:
    """
    Class to store the output of a featurization run.
    """
    def __init__(self,
            data: np.ndarray,
            labels: List[str],
            featurization: str = None):
        self.data = data
        self.labels = labels
        self.featurization = featurization


########################
# Main analysis class. #
########################

allowed_formats = ("pdb", "pdb_dir", "traj", )

class EnsembleAnalysis:
    """
    TODO.
    """

    def __init__(self):
        # Store mdtraj trajectories here.
        self.ensembles = OrderedDict()
        self.refresh_all()
        
    #----------------------------------
    # Load data.                      -
    #----------------------------------

    def load_ensembles(
            self,
            input_dp: str,
            codes: List[str],
            format: str = "pdb",
            topology_fp: str = None,
            ext: str = None) -> None:
        """
        Load the xyz data from coordinate files.

        `input_dp`: directory where all the data files for the ensembles are.
        `codes`: codes of the ensembles to analyze. Those must be the basenames
            of the PDB files in 'pdb_dp'.
        `format`: can be one of the followings. `pdb` for a single PDB files
            with multiple models, `pdb_dir` for a directory with a list of PDB
            files of the same system (which correspond to different
            conformations), `traj` for a trajectory file (accepted ones are:
            xtc, dcd).
        `topology_fp`: topology file, takes only effect if `format` is set to
            `traj`.
        `ext`: extension of the input files. If not provided, it will be derived
            automatically.
        """

        # Check the input.
        self._check_append()

        if not format in allowed_formats:
            raise KeyError(f"Invalid 'format': {format}")

        if format == "traj":
            if topology_fp is None:
                raise ValueError(
                    "Must provide a 'topology_fp' when 'format' is 'traj'.")

        # Load the ensemble data.
        for code_i in codes:
            st.write(f"Loading data for {code_i}.")

            # TODO: convert to ensemble.
            if format == "pdb":
                traj_i = read_pdb(
                    input_fp=os.path.join(input_dp, f"{code_i}.pdb")

                )
            elif format == "pdb_dir":
                traj_i = read_pdb_dir(
                    input_dp=input_dp,
                    code=code_i
                )
            elif format == "traj":
                traj_i = read_traj(
                    input_dp=input_dp,
                    code=code_i,
                    topology_fp=topology_fp,
                    ext=ext
                )
            else:
                raise KeyError(format)
            
            if format in ("pdb", "pdb_dir"):
                # Save as DCD file (binary format) for faster loading next time.
                # st.write("- Saving a DCD file for faster reading next time.")
                # self.trajectories[code_i].save(ens_dcd_i)
                # self.trajectories[code_i][0].save(ens_top_i)
                pass
            
            # Add an ensemble object.
            ensemble_i = Ensemble(traj=traj_i, code=code_i)
            self.add_ensemble(ensemble_i)
            st.write(f"Found {len(self.ensembles[code_i])} conformations.")


    def add_ensemble(self, ensemble: Ensemble) -> None:
        """
        `ensemble`: an 'Ensemble' object.
        """
        self._check_append()
        self.ensembles[ensemble.code] = ensemble
    
    def _check_append(self) -> None:
        if self.featurized_data is not None:
            raise ValueError(
                "Can not add new ensembles when featurization was already"
                " performed. Use the 'refresh_all' to remove featurized data"
                " and then add a new ensemble.")
    

    def __getitem__(self, code: str) -> Ensemble:
        return self.ensembles[code]

    
    def loc(self, idx: int) -> Ensemble:
        for i, c in enumerate(self.codes):
            if i == idx:
                return self.ensembles[c]
        raise IndexError(i)
    
    def get_num_residues(self) -> int:
        top = self.loc(0).traj.topology
        residues = [r for r in top.residues]
        return len(residues)

    @property
    def codes(self) -> list:
        """
        Look at the 'ensembles' objects and return the list of codes.
        """
        return list(self.ensembles.keys())

    @property
    def trajectories(self) -> dict:
        """
        Look at the 'ensembles' objects and return a dictionary with
        mdtraj trajectories as values.
        """
        return {code: self.ensembles[code].traj for code in self.ensembles}
    
    @property
    def labels(self) -> np.ndarray:
        """
        If we have 2 ensembles with 3 frames each, the 'concat_ids' array will
        look like:
            [0, 0, 0, 1, 1, 1]
        """
        concat_labels = []
        for i, code_i in enumerate(self.codes):
            ids_i = np.full(len(self.ensembles[code_i]), i)
            concat_labels.append(ids_i)
        return np.concatenate(concat_labels, axis=0)
    
    def __len__(self) -> int:
        """
        Rreturns the number of ensembles currently loaded.
        """
        return len(self.ensembles)


    #----------------------------------
    # Featurize ensembles.            -
    #----------------------------------

    def featurize(self,
            featurization: Union[str, Callable],
            params: dict = {}) -> None:
        """
        Featurize the ensembles.

        `featurization`: name of the featurization scheme built-in in DPET.
            Alternatively, a callable (for a user-defined function).
        `params`: a dictionary with the parameters for the featurization
            function.
        """

        # Check input.
        if not self.ensembles:
            raise ValueError(
                "No ensemble was loaded. Use the 'load_ensembles' or"
                "'add_ensemble' methods to load some structures first.")

        _params = params.copy()
        if "normalize" in _params:
            if featurization not in ("ca_dist",):
                raise ValueError()
            _params.pop("normalize")
            use_norm = params["normalize"]
        else:
            use_norm = False

        # We will store featurized data for the ensembles in featurized_data dictioanry.
        featurized_data = OrderedDict()

        for i, code_i in enumerate(self.codes):
            st.write(f"Featurizing the {code_i} ensemble.")
            feat_out = self.ensembles[code_i].featurize(
                featurization=featurization,
                get_names=i == 0,
                **_params,
            )

            if i == 0:  # When processing the first ensemble, we also get the feature names.
                featurized_data[code_i], feature_names = feat_out
            else:
                featurized_data[code_i] = feat_out
            st.write("Featurized ensemble shape:",
                featurized_data[code_i].shape)

        features_concat = self._get_concat_data(
            data=featurized_data, codes=None
        )

        if use_norm:
            if featurization == "ca_dist":
                mean = features_concat.mean(axis=0)
                std = features_concat.std(axis=0)
                features_concat = (features_concat - mean) / std
                for i, code_i in enumerate(self.codes):
                    featurized_data[code_i] = (
                        featurized_data[code_i] - mean
                    ) / std
            else:
                raise NotImplementedError()
        
        self.featurized_data = featurized_data
        self.features_concat = features_concat
        self.feature_names = feature_names
        self.featurization = featurization
        self.featurization_params = params


    def refresh_features(self):
        """
        Inititialize the featurized data. Also used to clean from memory the
        featurized data if not needed anymore.
        """
        self.featurized_data = None
        self.features_concat = None
        self.feature_names = None
        self.featurization = None
        self.featurization_params = None

    def refresh_all(self):
        """
        Refresh everything.
        """
        self.refresh_features()


    # def rg_calculator(self):
    #     rg_values_list = []
    #     for traj in self.trajectories.values():
    #         rg_values_list.extend(calculate_rg_for_trajectory(traj))
    #     return [item[0] * 10 for item in rg_values_list]


    # Only used for all ensembles.
    def get_features(
            self,
            featurization: str,
            codes: Union[str, List[str]] = None,
            concat: bool = False,
            params: dict = {}) -> Union[np.ndarray, dict]:
        """
        TODO: important documenation here. This is the method that users should
        call to extract features that they want to analyze.
        """

        _codes = self._get_codes(codes)[0]

        featurized_data = OrderedDict()

        for i, code_i in enumerate(_codes):
            st.write(f"Calculating {featurization} for {code_i}.")
            feat_out = self.ensembles[code_i].featurize(
                featurization=featurization,
                get_names=False,
                ravel=False,
                **params,
            )
            featurized_data[code_i] = feat_out
        
        if concat:
            features_concat = self._get_concat_data(
               data=featurized_data, codes=_codes
            )
            return features_concat
        else:
            return featurized_data


    def get_phi_psi(self, codes: List[str] = None, concat: bool = False):
        return self.get_features(
            featurization="phi_psi",
            codes=codes,
            concat=concat
        )


    def get_rg(self, codes: List[str] = None, concat: bool = False):
        return self.get_features(
            featurization="rg",
            codes=codes,
            concat=concat
        )


    #----------------------------------
    # Dimensionality reduction.       -
    #----------------------------------

    def dimensionality_reduction(
            self,
            method: str,
            params: dict = {},
            fit_on: list = None) -> None:
        """
        Perform actual dimensionality reduction.

        TODO: more documentation about the arguments.
        `fit_on`: list of the ensembles codes on which to fit the dimensionality
            reduction method. If 'None' it will be fit on a concatenation all
            the ensembles.
        """

        ## Process the input.

        # Check the input.
        if self.featurized_data is None:
            raise ValueError(
                "No featurization was performed. Use the 'featurize' method to"
                " extract some features from the ensembles first")
        
        fit_on, _all_ensembles = self._get_codes(fit_on)

        if _all_ensembles:  # Fit on all ensembles.
            fit_on = self.codes
        else:  # Fit only on some user-defined ensembles.
            if method not in ("pca", "kpca"):
                raise ValueError(
                    f"Can not perform a '{method}' dimensionality reduction on"
                    " only subset of the ensembles. Set the 'fit_on' argument"
                    " as None or 'all'")
        
        if params.get("circular"):
            # Check if dimensionality reduction for circular data can be
            # performed.
            if method not in ("tsne", "kpca"):
                raise ValueError(
                    f"{method} is not adapted for angular data in DPET")
            if self.featurization not in ("phi_psi", "tr_omega", "tr_phi"):
                raise ValueError(
                    f"The current features ({self.featurization}) are not"
                    f" angular values. Can not perform {method} for circular"
                    " data")

        # Prepare the input. Concatenate the data along the "snapshot" axis. We
        # obtain a big ensemble with featurized representations from original
        # ensembles.
        concat_features = self._get_concat_data(
            data=self.featurized_data, codes=fit_on
        )
        
        
        ## Perform dimensionality reduction.

        st.write(
            "Performing dimensionality reduction on"
            f" {self.featurization} data.")
        st.write(
            "Input featurized ensemble shape:", concat_features.shape)
        st.write(f"Fitting...")

        reduce_dim_data = OrderedDict()
        reduce_dim_model = DimensionalityReduction(method=method)

        # Dimensionality reduction methods that can be fitted on only a part of
        # the ensembles.
        if method in ("pca", "kpca"):
            
            # Fit PCA on a set of selected ensembles.
            reduce_dim_model.fit(features=concat_features, **params)

            # Transform all ensembles via PCA.
            for code_i in self.codes:
                st.write(f"Transforming {code_i}.")
                reduced_ens_i = reduce_dim_model.transform(
                    features=self.featurized_data[code_i],
                )
                reduce_dim_data[code_i] = reduced_ens_i
                st.write(
                    "Reduced dimensionality ensemble shape:",
                    reduce_dim_data[code_i].shape,
                )

            # Concatenate all the ensembles.
            concat_reduce_dim_data = self._get_concat_data(
                data=reduce_dim_data, codes=None
            )

        # Dimensionality reduction methods that must be fitted on all the
        # ensembles.
        elif method in ("tsne", ):  # "dimenfix"

            # Fit the dimensionality reduction model and transform all input
            # data. Store the results in a single, concatenated array.
            rd_out = reduce_dim_model.fit(
                features=concat_features, **params)
            
            # Optionally stores a KMeans object used to select the right
            # perplexity value.
            if "kmeans" in rd_out:
                reduce_dim_model.set_clusters(
                    clst=rd_out["kmeans"],
                    name="kmeans"
                )
            
            # Store the tsne features for the concatenated ensemble.
            concat_reduce_dim_data = rd_out["data"]

            # Obtain the transformed features for each of the input ensembles.
            count = 0
            for code_i in fit_on:
                size_i = self.featurized_data[code_i].shape[0]
                reduced_ens_i = concat_reduce_dim_data[count:count+size_i]
                reduce_dim_data[code_i] = reduced_ens_i
                count += size_i
        
        else:
            raise KeyError(method)

        ## Return results.
        reduce_dim_model.set_data(
            data=reduce_dim_data,
            concat_data=concat_reduce_dim_data
        )
        
        return reduce_dim_model


    #--------------------
    # Internal methods. -
    #--------------------

    def _get_concat_data(
            self,
            data: dict,
            codes: Union[str, List[str]] = None,
            get_ids: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        TODO: add more documentation.

        arguments: ...

        returns: ...
        """

        _codes = self._get_codes(codes)[0]
        
        # Concatenate conformations from multiple ensembles. 
        concat_data = np.concatenate(
            [data[code_i] for code_i in _codes], axis=0
        )

        # Return the concat features and an array with the ids of their
        # ensembles.
        if get_ids:
            # concat_ids = []
            # for i, code_i in enumerate(self.codes):
            #     if code_i in _codes:
            #         ids_i = np.full(data[code_i].shape[0], i)
            #         concat_ids.append(ids_i)
            # concat_ids = np.concatenate(concat_ids, axis=0)
            # return concat_data, concat_ids
            raise NotImplementedError(
                "Keep the code for now, could be used later?")

        # Return only the concat features.
        else:
            return concat_data


    def _get_codes(self, codes: Union[str, List[str]]):

        if codes is None:  # Use all codes if none are provided.
            return (self.codes, True)
        elif isinstance(codes, str):  
            if codes == "all":  # Use all codes if "all" is provided.
                return (self.codes, True)
            else:
                raise TypeError()
        else:
            if isinstance(codes, (tuple, list, set, np.ndarray)):
                # Use user-defined codes.
                for c in codes:
                    if c not in self.codes:  # Check every code.
                        raise KeyError(
                            f"Code {c} not found among the codes for the"
                            " ensembles that are currently loaded")
                return (codes, False)
            else:
                raise TypeError()
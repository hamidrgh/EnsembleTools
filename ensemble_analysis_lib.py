import os
import numpy as np
import sklearn
import mdtraj



########################
# Main analysis class. #
########################

class EnsembleAnalysis:
    
    def __init__(self,
                 pdb_dp: str,
                 ens_codes: list):
        """
        Initialize and load the xyz data from PDB files.

        pdb_dp: directory where all the PDB files with the ensembles are.
        ens_codes: codes of the ensembles to analyze. Those must be the basenames
            of the PDB files in 'pdb_dp'.
        """

        self.ens_codes = ens_codes
        
        # Store mdtraj trajectories here.
        self.trajectories = {}
        
        for code_i in ens_codes:

            ens_dcd_i = os.path.join(pdb_dp, f"{code_i}.dcd") 
            ens_top_i = os.path.join(pdb_dp, f"{code_i}.top.pdb")

            if not os.path.isfile(ens_dcd_i): # check whether the specified path is an existing regular file or not
                ens_fp_i = os.path.join(pdb_dp, f"{code_i}.pdb")
                print(f"# Loading {ens_fp_i}.")
                self.trajectories[code_i] = mdtraj.load(ens_fp_i)
                # Save as DCD file (binary format) for faster loading next time.
                print("- Saving a DCD file for faster reading next time.")
                self.trajectories[code_i].save(ens_dcd_i)
                self.trajectories[code_i][0].save(ens_top_i)
            else:
                print(f"# Loading {ens_dcd_i}.")
                self.trajectories[code_i] = mdtraj.load(ens_dcd_i, top=ens_top_i)

            print(f"- Found {len(self.trajectories[code_i])} conformations.")
    

    def featurize(self,
                  featurization: str,
                  featurization_params: dict = {}):
        """
        Featurize the ensembles.
        """

        # Normalization options.
        # featurization_params = {"ca_dist": {"seq_sep": 2, "normalize": True},
        #                         "phi_psi": {},
        #                         "a_angle": {}}

        _featurization_params = featurization_params.copy()
        if "normalize" in _featurization_params:
            if featurization not in ("ca_dist", ):
                raise ValueError()
            _featurization_params.pop("normalize")
            use_norm = featurization_params["normalize"]
        else:
            use_norm = False

        # We will store featurized data for the ensembles in featurized_data dictioanry.
        # A list of labels is built here to label the clusters made by t-sne and kmeans clustering
        self.all_labels = []
        self.featurized_data = {}
        for i, code_i in enumerate(self.ens_codes):
            print(f"# Featurizing the {code_i} ensemble.")
            feat_out = featurize(traj=self.trajectories[code_i],
                                featurization=featurization,
                                get_names=i == 0,
                                **_featurization_params)

            
            if i == 0:  # When processing the first ensemble, we also get the feature names.
                self.featurized_data[code_i], self.feature_names = feat_out
            else:
                self.featurized_data[code_i] = feat_out
            print("- Featurized ensemble shape:",
                  self.featurized_data[code_i].shape)
        
        # Concatenate the data along the "snapshot" axis. We basically obtain a big ensemble with
        # featurized representations from all the original ensembles.
        concat_features = []
        for code_i in self.ens_codes:
            concat_features.append(self.featurized_data[code_i])
        self.concat_features = np.concatenate(concat_features, axis=0)
        print("Concatenated featurized ensemble shape:",
              self.concat_features.shape)
        
        for label, data_points in self.featurized_data.items():
            num_data_points = len(data_points)
            self.all_labels.extend([label] * num_data_points)

        if use_norm:
            if featurization == "ca_dist":
                mean = self.concat_features.mean(axis=0)
                std = self.concat_features.std(axis=0)
                self.concat_features = (self.concat_features-mean)/std
                for i, code_i in enumerate(self.ens_codes):
                    self.featurized_data[code_i] = (self.featurized_data[code_i]-mean)/std
            else:
                raise NotImplementedError()

    def rg_calculator(self):
        rg_values_list = []
        for traj in self.trajectories.values():
            rg_values_list.extend(calculate_rg_for_trajectory(traj))
        return [item[0]*10 for item in rg_values_list]
    

    def fit_dimensionality_reduction(self,
                                     reduce_dim_method: str,
                                     reduce_dim_params: dict = {}):
        # reduce_dim_params = {"pca": {"num_dim": None}}
        # Perform actual dimensionality reduction.
        # adding the tsne part here (fir_transform) using if statement 
        if reduce_dim_method == 'pca':
            self.reduce_dim_model = fit_dimensionality_reduction(
                self.concat_features,
                method=reduce_dim_method,
                **reduce_dim_params)

            # Transform the ensembles using the dimensionality reduction method.
            
            self.reduce_dim_data = {}
            for code_i in self.ens_codes:
                print(f"# Transforming {code_i}.")
                self.reduce_dim_data[code_i] = transform(
                    features=self.featurized_data[code_i],
                    model=self.reduce_dim_model,
                    method=reduce_dim_method)
                print(f"- Reduced dimensionality ensemble shape:",
                    self.reduce_dim_data[code_i].shape)

            self.concat_reduce_dim_data = transform(
                features=self.concat_features,
                model=self.reduce_dim_model,
                method=reduce_dim_method)
        elif reduce_dim_method == 'tsne':
                self.reduce_dim_model = fit_dimensionality_reduction(
                self.concat_features,
                method=reduce_dim_method,
                **reduce_dim_params)
        elif reduce_dim_method == "dimenfix":
                self.reduce_dim_model = fit_dimensionality_reduction(
                self.concat_features, method=reduce_dim_method, **reduce_dim_params
                )
        
        elif reduce_dim_method == 'tsne-circular':
            self.reduce_dim_model = fit_dimensionality_reduction(
            self.concat_features,
            method=reduce_dim_method,
            **reduce_dim_params)

                
        else:
            raise NotImplementedError()

                
        

######################################
# Featurization of input structures. #
######################################

def featurize(traj, featurization, get_names=False, *args, **kwargs):
    """
    Featurize a whole ensemble, whose xyz coordinates are stored in
    the 'traj' object.
    """
    if featurization == "ca_dist":
        return featurize_ca_dist(traj, get_names=get_names, *args, **kwargs)
    elif featurization == "phi_psi":
        return featurize_phi_psi(traj, get_names=get_names, *args, **kwargs)
    elif featurization == "a_angle":
        return featurize_a_angle(traj, get_names=get_names, *args, **kwargs)
    else:
        raise NotImplementedError(
            "Other featurizations could be implemented here.")


def featurize_ca_dist(traj, seq_sep=2, inverse=False, get_names=False):
    # Get all C-alpha indices.
    ca_ids = traj.topology.select("name == CA")
    atoms = list(traj.topology.atoms)
    # Get all pair of ids.
    pair_ids = []
    names = []
    for i, id_i in enumerate(ca_ids):
        for j, id_j in enumerate(ca_ids):
            if j - i >= seq_sep:
                pair_ids.append([id_i, id_j])
                if get_names:
                    names.append(
                        f"{repr(atoms[id_i].residue)}-{repr(atoms[id_j].residue)}")
    # Calculate C-alpha - C-alpha distances.
    ca_dist = mdtraj.compute_distances(traj=traj, atom_pairs=pair_ids)
    if inverse:
        ca_dist = 1/ca_dist
    if get_names:
        return ca_dist, names
    else:
        return ca_dist

def rg_calculator(traj):
    return mdtraj.compute_rg(traj)

def calc_ca_dmap(traj):
    ca_ids = traj.topology.select("protein and name CA")
    ca_xyz = traj.xyz[:,ca_ids]
    dmap = np.sqrt(np.sum(np.square(ca_xyz[:,None,:,:] - ca_xyz[:,:,None,:]), axis=3))
    return dmap

def featurize_phi_psi(traj, get_names=False):
    atoms = list(traj.topology.atoms)
    phi_ids, phi = mdtraj.compute_phi(traj)
    psi_ids, psi = mdtraj.compute_psi(traj)
    phi_psi = np.concatenate([phi, psi], axis=1)
    if get_names:
        names = []
        for t in phi_ids:
            names.append(repr(atoms[t[1]].residue) + "-PHI")
        for t in psi_ids:
            names.append(repr(atoms[t[0]].residue) + "-PSI")
        return phi_psi, names
    else:
        return phi_psi
    

def featurize_a_angle(traj, get_names=False):
    # Get all C-alpha indices.
    ca_ids = traj.topology.select("protein and name CB")
    atoms = list(traj.topology.atoms)
    # Get all pair of ids.
    tors_ids = []
    names = []
    for i in range(ca_ids.shape[0]-3):
        if get_names:
            names.append("{}:{}:{}:{}".format(atoms[ca_ids[i]],
                                              atoms[ca_ids[i+1]],
                                              atoms[ca_ids[i+2]],
                                              atoms[ca_ids[i+3]]))
        tors_ids.append((ca_ids[i], ca_ids[i+1], ca_ids[i+2], ca_ids[i+3]))
    tors = mdtraj.compute_dihedrals(traj, tors_ids)
    if get_names:
        return tors, names
    else:
        return tors

#######################
# t-SNE fro circular data. #
#######################


def unit_vectorize(a):
    """Convert an array with (*, N) angles in an array with (*, N, 2) sine and
    cosine values for the N angles."""
    v = np.concatenate([np.cos(a)[...,None], np.sin(a)[...,None]], axis=-1)
    return v

def unit_vector_distance(a0, a1):
    """Compute the sum of distances between two (*, N, 2) arrays storing the
    sine and cosine values of N angles."""
    v0 = unit_vectorize(a0)
    v1 = unit_vectorize(a1)
    # Distance between every pair of N angles.
    dist = np.sqrt(np.square(v0 - v1).sum(axis=-1))
    # We sum over the N angles.
    dist = dist.sum(axis=-1)
    return dist

#############################
# Dimensionality reduction. #
#############################

def fit_dimensionality_reduction(concat_features, method="pca", *args, **params):
    if method == "pca":
        return fit_pca(concat_features, *args, **params)
    elif method == "tsne":
        print('tsne is your selected dimensionality reduction method!')
        return fit_transform_tsne(concat_features, *args, **params)
    elif method == "dimenfix":
        print("dimenfix is your selected dimensionality reduction method!")
        return fit_transform_dimenfix(concat_features, *args, **params ) 
    elif method == "mds":
        return fit_transform_mds(concat_features, *args, **params)
    elif method == "tsne-circular":
        return fit_transform_tsne_circular(concat_features, *args, **params)
    else:
        raise NotImplementedError(
            "Other dimensionality reduction methods could be implemented here.")
        
def fit_pca(data, num_dim):
    pca = sklearn.decomposition.PCA(n_components=num_dim)
    pca.fit(data)
    return pca

def fit_transform_tsne(data, perplexityVals= range(2,10,2) , metric = 'euclidean', dir='.'):
    from sklearn.manifold import TSNE
    print('tsne is running...')
    for i in perplexityVals:
        tsneObject = TSNE(n_components=2, perplexity=i, early_exaggeration=10.0, learning_rate=100.0, n_iter=3500, metric = metric ,n_iter_without_progress=300, min_grad_norm=1e-7,  init='random', method='barnes_hut', angle=0.5)
        tsne = tsneObject.fit_transform(data)
        np.savetxt( dir + "/tsnep{0}".format(i), tsne)
        print(f'tsne file for the perplexity value of {i} is saved in {dir} ')
    print(f'tsne is done! All files saved in {dir}')

def fit_transform_tsne_circular(data, perplexityVals= range(2,10,2) , metric = unit_vector_distance, dir='.'):
    from sklearn.manifold import TSNE
    print('tsne for phi_psi is running...')
    for i in perplexityVals:
        tsneObject = TSNE(n_components=2, perplexity=i, early_exaggeration=10.0, learning_rate=100.0, n_iter=3500, metric = metric ,n_iter_without_progress=300, min_grad_norm=1e-7,  init='random', method='barnes_hut', angle=0.5)
        tsne = tsneObject.fit_transform(data)
        np.savetxt( dir + "/tsnep{0}".format(i), tsne)
        print(f'tsne file for the perplexity value of {i} is saved in {dir} ')
    print(f'tsne is done! All files saved in {dir}')
    pass
   
def fit_transform_dimenfix(data):
    from neo_force_scheme import NeoForceScheme
    nfs = NeoForceScheme()
    projection = nfs.fit_transform(data)
    return projection

def fit_transform_mds(data, nume_dim = 2):
    from sklearn.manifold import MDS
    embedding = MDS(n_components= nume_dim)
    feature_transformed = embedding.fit_transform(data)
    return feature_transformed
    

def transform(features, model, method="pca"):
    if method == "pca":
        return model.transform(features)
    else:
        raise NotImplementedError(
            "Other transformation methods could be implemented here")



#######################
# Score similarities. #
#######################

def score_kld_approximation(v_true, v_pred, n_bins=50, pseudo_c=0.001):
    """
    Scores an approximation of KLD by discretizing the values in
    'v_true' (data points from a reference distribution) and 'v_pred'
    (data points from a predicted distribution).
    """
    # Define bins.
    _min = min((v_true.min(), v_pred.min()))
    _max = max((v_true.max(), v_pred.max()))
    bins = np.linspace(_min, _max, n_bins+1)
    # Compute the frequencies in the bins.
    ht = (np.histogram(v_true, bins=bins)[0]+pseudo_c)/v_true.shape[0]
    hp = (np.histogram(v_pred, bins=bins)[0]+pseudo_c)/v_pred.shape[0]
    kl = -np.sum(ht*np.log(hp/ht))
    return kl, bins

def score_jsd_approximation(v_true, v_pred, n_bins=50, pseudo_c=0.001):
    """
    Scores an approximation of JS by discretizing the values in
    'v_true' (data points from a reference distribution) and 'v_pred'
    (data points from a predicted distribution).
    """
    # Define bins.
    _min = min((v_true.min(), v_pred.min()))
    _max = max((v_true.max(), v_pred.max()))
    bins = np.linspace(_min, _max, n_bins+1)
    # Compute the frequencies in the bins.
    ht = (np.histogram(v_true, bins=bins)[0]+pseudo_c)/v_true.shape[0]
    hp = (np.histogram(v_pred, bins=bins)[0]+pseudo_c)/v_pred.shape[0]
    hm = (ht + hp)/2
    kl_tm = -np.sum(ht*np.log(hm/ht))
    kl_pm = -np.sum(hp*np.log(hm/hp))
    js = 0.5*kl_pm + 0.5*kl_tm
    return js, bins

def score_akld_d(traj_ref: mdtraj.Trajectory,
                 traj_hat: mdtraj.Trajectory,
                 n_bins: int = 50,
                 method: str = "js"):
    """
    See the idpGAN article.
    """
    # Calculate distance maps.
    dmap_ref = calc_ca_dmap(traj_ref)
    dmap_hat = calc_ca_dmap(traj_hat)
    if dmap_ref.shape[1] != dmap_hat.shape[1]:
        raise ValueError(
            "Input trajectories have different number of residues:"
            f" ref={dmap_ref.shape[1]}, hat={dmap_hat.shape[1]}")
    n_akld_d = []
    if method == "kl":
        score_func = score_kld_approximation
    elif method == "js":
        score_func = score_jsd_approximation
    else:
        raise KeyError(method)
    for i in range(dmap_ref.shape[1]):
        for j in range(dmap_ref.shape[1]):
            if i+1 >= j:
                continue
            kld_d_ij = score_func(
                dmap_ref[:,i,j], dmap_hat[:,i,j],
                n_bins=n_bins)[0]
            n_akld_d.append(kld_d_ij)
    return np.mean(n_akld_d), n_akld_d


def score_akld_t(traj_ref: mdtraj.Trajectory,
                 traj_hat: mdtraj.Trajectory,
                 n_bins: int = 50,
                 method: str = "js"):
    """
    Similar to 'score_akld_d', but evaluate alpha torsion angles.
    """
    # Calculate distance maps.
    tors_ref = featurize_a_angle(traj_ref)
    tors_hat = featurize_a_angle(traj_hat)
    if tors_ref.shape[1] != tors_hat.shape[1]:
        raise ValueError(
            "Input trajectories have different number torsion angles:"
            f" ref={tors_ref.shape[1]}, hat={tors_hat.shape[1]}")
    n_akld_t = []
    if method == "kl":
        score_func = score_kld_approximation
    elif method == "js":
        score_func = score_jsd_approximation
    else:
        raise KeyError(method)
    for i in range(tors_ref.shape[1]):
        score_i = score_func(
            tors_ref[:,i], tors_hat[:,i],
            n_bins=n_bins)[0]
        n_akld_t.append(score_i)
    return np.mean(n_akld_t), n_akld_t


def calculate_rg_for_trajectory(trajectory):
    return [mdtraj.compute_rg(frame) for frame in trajectory]



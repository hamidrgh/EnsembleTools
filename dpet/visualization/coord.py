import numpy as np
import torch
import mdtraj

def get_distance_matrix(xyz):
    """
    Gets an ensemble of xyz conformations with shape (N, L, 3) and
    returns the corresponding distance matrices with shape (N, L, L).
    """
    return np.sqrt(np.sum(np.square(xyz[:,None,:,:]-xyz[:,:,None,:]), axis=3))

def get_contact_map(dmap, threshold=0.8, pseudo_count=0.01):
    """
    Gets a trajectory of distance maps with shape (N, L, L) and
    returns a (L, L) contact probability map.
    """
    n = dmap.shape[0]
    cmap = ((dmap <= threshold).astype(int).sum(axis=0)+pseudo_count)/n
    return cmap


def torch_chain_dihedrals(xyz, norm=False):
    r_sel = xyz
    b0 = -(r_sel[:,1:-2,:] - r_sel[:,0:-3,:])
    b1 = r_sel[:,2:-1,:] - r_sel[:,1:-2,:]
    b2 = r_sel[:,3:,:] - r_sel[:,2:-1,:]
    b0xb1 = torch.cross(b0, b1)
    b1xb2 = torch.cross(b2, b1)
    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2)
    y = torch.sum(b0xb1_x_b1xb2*b1, axis=2)*(1.0/torch.linalg.norm(b1, dim=2))
    x = torch.sum(b0xb1*b1xb2, axis=2)
    dh_vals = torch.atan2(y, x)
    if not norm:
        return dh_vals
    else:
        return dh_vals/np.pi
    

def create_consecutive_indices_matrix(ca_indices):

    """This function gets the CA indices of (L,) shape and 
    create all possible 4 consecutive indices with the shape (L-3, 4) """

    n = len(ca_indices)
    if n < 4:
        raise ValueError("Input array must contain at least 4 indices.")

    # Calculate the number of rows in the resulting matrix
    num_rows = n - 3

    # Create an empty matrix to store the consecutive indices
    consecutive_indices_matrix = np.zeros((num_rows, 4), dtype=int)

    # Fill the matrix with consecutive indices
    for i in range(num_rows):
        consecutive_indices_matrix[i] = ca_indices[i:i+4]

    return consecutive_indices_matrix


def calculate_asphericity(gyration_tensors):
    asphericities = []
    for gyration_tensor in gyration_tensors:
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(gyration_tensor)
        
        # Sort eigenvalues in ascending order
        eigenvalues.sort()
        
        # Calculate asphericity
        lambda_max = eigenvalues[-1]
        lambda_mid = eigenvalues[1]  # Middle eigenvalue
        lambda_min = eigenvalues[0]
        
        asphericity = (lambda_max - lambda_min) / (lambda_max + lambda_mid + lambda_min)
        asphericities.append(asphericity)
    
    return asphericities


def calculate_prolateness(gyration_tensors):
    prolateness_values = []
    for gyration_tensor in gyration_tensors:
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(gyration_tensor)
        eigenvalues.sort()  # Sort eigenvalues in ascending order

        # Calculate prolateness
        lambda_max = eigenvalues[-1]
        lambda_mid = eigenvalues[1]
        lambda_min = eigenvalues[0]

        prolateness = (lambda_mid - lambda_min) / lambda_max
        prolateness_values.append(prolateness)
    
    return prolateness_values

def contact_probability_map(traj, threshold = 0.8):
    distances = mdtraj.compute_contacts(traj, scheme="ca")[0]
    res_pair = mdtraj.compute_contacts(traj, scheme="ca")[1]
    contact_distance = mdtraj.geometry.squareform(distances, res_pair)
    matrix_contact_prob = np.zeros(contact_distance.shape)
    threshold = threshold
    for ens in range(contact_distance.shape[0]):
        for i in range(contact_distance.shape[1]):
            for j in range(contact_distance.shape[2]):
                if contact_distance[ens][i][j] < threshold:
                    matrix_contact_prob[ens][i][j] = 1
                else:
                    matrix_contact_prob[ens][i][j] = 0
    matrix_prob_avg = np.mean(matrix_contact_prob, axis=0)
    return matrix_prob_avg

def dict_phi_psi_normal_cases(dict_phi_psi):
    dict_phi_psi_normal_case={}
    for key in dict_phi_psi.keys():
        array_phi=dict_phi_psi[key][0][:,:-1]
        array_psi=dict_phi_psi[key][1][:,1:]
        dict_phi_psi_normal_case[key]=[array_phi,array_psi]
    return dict_phi_psi_normal_case

def split_dictionary_phipsiangles(dictionary):
    dict_phi_psi={}
    for key, value in dictionary.items():
        num_columns=len(value[0])
        split_index=num_columns//2
        phi_list=value[:,:split_index]
        psi_list=value[:,split_index:]
        dict_phi_psi[key]=[phi_list,psi_list]
    return dict_phi_psi

def ss_measure_disorder(featurized_data:dict):
    
    """This function accepts the dictionary of phi-psi arrays
    which is saved in featurized_data attribute and as an output provide
    lexibility parameter for each residue in the ensemble
    Note: this function only works on phi/psi feature """

    f = {}
    R_square_dict = {}

    for key in dict_phi_psi_normal_cases(split_dictionary_phipsiangles(featurized_data)).keys():
        Rsquare_phi = []
        Rsquare_psi = []

        phi_array = dict_phi_psi_normal_cases(split_dictionary_phipsiangles(featurized_data))[key][0]
        psi_array = dict_phi_psi_normal_cases(split_dictionary_phipsiangles(featurized_data))[key][1]
        if isinstance(phi_array, np.ndarray) and phi_array.ndim == 2:
            for i in range(phi_array.shape[1]):
                Rsquare_phi.append(round(np.square(np.sum(np.fromiter(((1 / phi_array.shape[0]) * np.cos(phi_array[c][i]) for c in range(phi_array.shape[0])), dtype=float))) + \
                        np.square(np.sum(np.fromiter(((1 / phi_array.shape[0]) * np.sin(phi_array[c][i]) for c in range(phi_array.shape[0])), dtype=float))),5))

        if isinstance(psi_array, np.ndarray) and psi_array.ndim == 2:
                for j in range(psi_array.shape[1]):
                    Rsquare_psi.append(round(np.square(np.sum(np.fromiter(((1 / psi_array.shape[0]) * np.cos(psi_array[c][j]) for c in range(psi_array.shape[0])), dtype=float))) + \
                          np.square(np.sum(np.fromiter(((1 / psi_array.shape[0]) * np.sin(psi_array[c][j]) for c in range(psi_array.shape[0])), dtype=float))),5))


        R_square_dict[key] = [Rsquare_phi, Rsquare_psi]

    for k in R_square_dict.keys():
        f_i=[]
        for z in range(len(R_square_dict[k][0])):
            f_i.append(round(1 - (1/2 * np.sqrt(R_square_dict[k][0][z])) - (1/2 * np.sqrt(R_square_dict[k][1][z])),5))
            f[k]=f_i
    return f
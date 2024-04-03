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
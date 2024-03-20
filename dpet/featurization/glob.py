import numpy as np
import mdtraj


def rg_calculator(traj: mdtraj.Trajectory) -> np.ndarray:
    return mdtraj.compute_rg(traj)



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
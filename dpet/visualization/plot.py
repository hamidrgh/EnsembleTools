import matplotlib.pyplot as plt
import numpy as np
from dpet.featurization.distances import calc_ca_dmap
import mdtraj



def plot_average_dmap(traj_dict, ticks_fontsize=14,
                                 cbar_fontsize=14,
                                 title_fontsize=14,
                                 dpi=96,
                                 max_d=6.8,
                                 use_ylabel=True):
    """
    Plot the average distance map comparison for multiple proteins.

    Parameters:
        ens_dict (dict): A dictionary where keys are protein names and values are their trajectory.
        ticks_fontsize (int): Font size for ticks. Default is 14.
        cbar_fontsize (int): Font size for color bar ticks. Default is 14.
        title_fontsize (int): Font size for title. Default is 14.
        dpi (int): Dots per inch, controlling the resolution of the resulting plot. Default is 96.
        max_d (float): Maximum distance value for color bar. Default is 6.8.
        use_ylabel (bool): Whether to use y-labels. Default is True.

    Returns:
        None
    """
    num_proteins = len(traj_dict)
    cols = 2  # Number of columns for subplots
    rows = (num_proteins + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
    
    for i, (protein_name, traj) in enumerate(traj_dict.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if num_proteins > 1 else axes
        ens_data = calc_ca_dmap(traj)
        avg_dmap = np.mean(ens_data, axis=0)
        tril_ids = np.tril_indices(avg_dmap.shape[0], 0)
        avg_dmap[tril_ids] = np.nan
        
        im = ax.imshow(avg_dmap)
        ax.set_title(f"Average Distance Map: {protein_name}", fontsize=title_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
        if not use_ylabel:
            ax.set_yticks([])
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Average $d_{ij}$ [nm]", fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        
        if max_d is not None:
            im.set_clim(0, max_d)
    
    # Remove any empty subplots
    for i in range(num_proteins, rows * cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout()
    plt.show()



def end_to_end_distances_plot(traj_dict, atom_selector ="protein and name CA", bins = 50):
    ca_indices = traj_dict[next(iter(traj_dict))].topology.select(atom_selector)
    for ens in traj_dict:
        plt.hist(mdtraj.compute_distances(traj_dict[ens],[[ca_indices[0], ca_indices[-1]]]).ravel()
                  , label=ens, bins=bins, edgecolor = 'black', density=True)
    plt.title("End-to-End distances distribution")
    plt.legend()
    plt.show()    


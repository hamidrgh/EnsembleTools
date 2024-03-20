import matplotlib.pyplot as plt
import numpy as np
from dpet.featurization.distances import calc_ca_dmap
from dpet.featurization.glob import *
from dpet.featurization.angles import *
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

def plot_asphericity_dist(dict_traj, bins = 50):
    for ens in dict_traj:
        asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(dict_traj[ens]))
        plt.hist(asphericity, label=ens, bins=bins, edgecolor = 'black', density=True)
    plt.legend()
    plt.show()

def plot_rg_vs_asphericity(dict_traj):
    for ens in dict_traj:
        x = rg_calculator(dict_traj[ens])
        y = calculate_asphericity(mdtraj.compute_gyration_tensor(dict_traj[ens]))
        plt.scatter(x, y , s =4, label=ens)
    plt.ylabel("Asphericity")
    plt.xlabel("Rg [nm]")
    plt.legend()
    plt.show()

def plot_prolateness_dist(dict_traj, bins = 50):
    for ens in dict_traj:
        prolat = calculate_prolateness(mdtraj.compute_gyration_tensor(dict_traj[ens]))
        plt.hist(prolat, label=ens, bins=bins, edgecolor = 'black', density=True)
    plt.legend()
    plt.show()

def plot_rg_vs_prolateness(dict_traj, bins=50):
    for ens in dict_traj:
        x = rg_calculator(dict_traj[ens])
        y = calculate_prolateness(mdtraj.compute_gyration_tensor(dict_traj[ens]))
        plt.scatter(x, y, s=4, label=ens)
    plt.ylabel("prolateness")
    plt.xlabel("Rg [nm]")
    plt.legend()
    plt.show()

def plot_alpha_angles_dist(dict_traj, bins =50):
    for ens in dict_traj:
        plt.hist(featurize_a_angle(dict_traj[ens])[0].ravel(), bins=bins, histtype="step", density=True, label=ens)
    plt.title("the distribution of dihedral angles between four consecutive Cα beads.")
    plt.legend()
    plt.show()

def plot_relative_helix_content(dict_traj):

    _dssp_data_dict = {}
    for ens in dict_traj:
        _dssp_data_dict[ens] = mdtraj.compute_dssp(dict_traj[ens])
    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(next(iter(_dssp_data_dict.values())).shape[1])

    for protein_name, dssp_data in _dssp_data_dict.items():

        h_count = np.count_nonzero(dssp_data == "H", axis=0)

        total_residues = dssp_data.shape[0]

        relative_h_content = h_count / total_residues
        ax.bar(range(len(relative_h_content)), relative_h_content, bottom=bottom, label=protein_name) 
        bottom += relative_h_content

    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Relative Content of H (Helix)')
    ax.set_title('Relative Content of H in Each Residue in the ensembles')
    ax.legend()
    plt.show()

def plot_rg_comparison(dict_traj, n_bins=50, bins_range=(1, 4.5), dpi=96 ):
    from matplotlib.lines import Line2D
    rg_dict = {}
    for ens in dict_traj:
        rg_dict[ens] = rg_calculator(dict_traj[ens])
    
    h_args = {"histtype": "step", "density": True}
    n_systems = len(rg_dict)
    bins = np.linspace(bins_range[0], bins_range[1], n_bins + 1)
    fig, ax = plt.subplots(1, n_systems, figsize=(3 * n_systems, 3), dpi=dpi)

    for i, (name_i, rg_i) in enumerate(rg_dict.items()):
        ax[i].hist(rg_i, bins=bins, label=name_i, **h_args)
        ax[i].set_title(name_i)
        if i == 0:
            ax[i].set_ylabel("Density")
        ax[i].set_xlabel("Rg [nm]")
        mean_rg = np.mean(rg_i)
        median_rg = np.median(rg_i)

        mean_line = ax[i].axvline(mean_rg, color='k', linestyle='dashed', linewidth=1)
        median_line = ax[i].axvline(median_rg, color='r', linestyle='dashed', linewidth=1)
    
    mean_legend = Line2D([0], [0], color='k', linestyle='dashed', linewidth=1, label='Mean')
    median_legend = Line2D([0], [0], color='r', linestyle='dashed', linewidth=1, label='Median')
    fig.legend(handles=[mean_legend, median_legend], loc='upper right')

    plt.tight_layout()
    plt.show()

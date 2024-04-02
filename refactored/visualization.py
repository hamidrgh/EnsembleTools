import os
import random
from matplotlib import cm, colors, legend, pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import plotly.graph_objects as go 
import plotly.express as px
import mdtraj
from matplotlib.lines import Line2D

from coord import calculate_asphericity, calculate_prolateness, contact_probability_map, create_consecutive_indices_matrix, get_contact_map, get_distance_matrix
from featurizer import FeaturizationFactory

def tsne_ramachandran_plot(tsne_kmeans_dir, concat_feature_phi_psi):
    s = np.loadtxt(tsne_kmeans_dir  +'/silhouette.txt')
    [bestP,bestK] = s[np.argmax(s[:,4]), 0], s[np.argmax(s[:,4]), 1]
    print([bestP,bestK])
    besttsne = np.loadtxt(tsne_kmeans_dir  + '/tsnep'+str(int(bestP)))
    best_kmeans = KMeans(n_clusters=int(bestK), n_init='auto').fit(besttsne)
    fig,axes = plt.subplots(1, 2, figsize = (10,5))

    for cluster_id, ax in zip(range(int(bestK)),axes.ravel()):
        cluster_frames = np.where(best_kmeans.labels_ == cluster_id)[0]
        print(cluster_frames, cluster_id)
        
        phi_psi_cluster_id = np.degrees(concat_feature_phi_psi[cluster_frames]).ravel()

        phi_flat = phi_psi_cluster_id[0::2]

        psi_flat = phi_psi_cluster_id[1::2]

        ax.scatter(phi_flat, psi_flat, alpha=0.5)

        ax.set_title(f'Ramachandran Plot for cluster {cluster_id}')
        ax.set_xlabel('Phi (ϕ) Angle (degrees)')
        ax.set_ylabel('Psi (ψ) Angle (degrees)')
        
    plt.tight_layout()
    plt.show()

def tsne_ramachandran_plot_density(tsne_dir, concat_features):
    s = np.loadtxt(tsne_dir  +'/silhouette.txt')
    [bestP,bestK] = s[np.argmax(s[:,4]), 0], s[np.argmax(s[:,4]), 1]
    print([bestP,bestK])
    besttsne = np.loadtxt(tsne_dir  + '/tsnep'+str(int(bestP)))
    best_kmeans = KMeans(n_clusters=int(bestK), n_init='auto').fit(besttsne)
    from matplotlib import colors
    rama_bins = 50
    rama_linspace = np.linspace(-180,180, rama_bins)
    fig,axes = plt.subplots(1, 2, figsize = (10,5))

    for cluster_id, ax in zip(range(int(bestK)),axes.flatten()):
        cluster_frames = np.where(best_kmeans.labels_ == cluster_id)[0]
        print(cluster_frames)
        
        phi_psi_cluster_id = np.degrees(concat_features[cluster_frames]).flatten()
        phi_flat = phi_psi_cluster_id[0::2]

        psi_flat = phi_psi_cluster_id[1::2]
        hist = ax.hist2d(phi_flat, psi_flat, cmap="viridis", bins=(rama_linspace, rama_linspace),  norm=colors.LogNorm(),density=True  )
        cbar = fig.colorbar(hist[-1], ax=ax)

        cbar.set_label("Density")
        
        ax.set_title(f'Ramachandran Plot for cluster {cluster_id}')
        ax.set_xlabel('Phi (ϕ) Angle (degrees)')
        ax.set_ylabel('Psi (ψ) Angle (degrees)')
    
    plt.tight_layout()
    plt.show()

def tsne_scatter_plot(tsne_dir, all_labels, ens_codes, rg):
    fig , (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14 ,4)) 
    s = np.loadtxt(tsne_dir  +'/silhouette.txt')
    [bestP,bestK] = s[np.argmax(s[:,4]), 0], s[np.argmax(s[:,4]), 1]
    besttsne = np.loadtxt(tsne_dir  + '/tsnep'+str(int(bestP)))
    bestclust = np.loadtxt(tsne_dir  +'/kmeans_'+str(int(bestK))+'clusters_tsnep'+str(int(bestP))+'.dat')
    print(bestP, bestK)
   

    # scatter original  labels
    label_colors = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in ens_codes}
    point_colors = list(map(lambda label: label_colors[label], all_labels))
    scatter_labeled = ax1.scatter(besttsne[:, 0], besttsne[:, 1], c=point_colors, s=10, alpha = 0.5)
    
    # scatter Rg labels 
    # Rg in Angstrom
    rg_labeled = ax3.scatter(besttsne[:, 0], besttsne[:, 1], c= [rg for rg in rg], s=10, alpha=0.5) 
    cbar = plt.colorbar(rg_labeled, ax=ax3)
    
    # scatter cluster labels
    cmap = cm.get_cmap('jet', bestK)
    scatter_cluster = ax2.scatter(besttsne[:,0], besttsne[:,1], c= bestclust.astype(float), s=10,cmap=cmap ,alpha=0.5)
    
    # manage legend
    legend_labels = list(label_colors.keys())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10) for label in legend_labels]
    fig.legend(legend_handles, legend_labels, title='Origanl Labels', loc = 'lower left')

    # KDE plot
    sns.kdeplot(x=besttsne[:, 0], y=besttsne[:, 1], ax=ax4, fill=True, cmap='Blues', levels=5)

    # ax1.scatter(grid_positions[0, densest_indices], grid_positions[1, densest_indices], c='red', marker='x', s=50, label='Densest Points')
    ax1.set_title('Scatter plot (original labels)')
    ax2.set_title('Scatter plot (clustering labels)')
    ax3.set_title('Scatter plot (Rg labels)')
    ax4.set_title('Density Plot ')
    
    plt.savefig(tsne_dir  +'/tsnep'+str(int(bestP))+'_kmeans'+str(int(bestK))+'.png', dpi=800)

def tsne_scatter_plot_2(tsne_dir, rg_numbers):
    s = np.loadtxt(tsne_dir  +'/silhouette.txt')
    [bestP,bestK] = s[np.argmax(s[:,4]), 0], s[np.argmax(s[:,4]), 1]
    besttsne = np.loadtxt(tsne_dir  + '/tsnep'+str(int(bestP)))
    print(bestP, bestK)

    row_numbers = np.arange(len(besttsne))
    fig1 = px.scatter(x=besttsne[:, 0], y=besttsne[:, 1], color=rg_numbers, labels={'color': 'Rg Labels'},
                    hover_data={'Row': row_numbers})
    fig1.show()

def dimenfix_scatter_plot(data, rg_numbers):
    fig = go.Figure(data=
                    go.Scatter(x=data[:, 0],
                                y=data[:, 1],
                                mode='markers',
                                
                                marker=dict(
                                    size=16,
                                    color= rg_numbers,
                                    colorscale='Viridis',
                                    showscale=True),
                                    
                            )
                    )
    fig.show()

def dimenfix_scatter_plot_2(data, all_labels):
    row = np.arange(len(data))
    fig = px.scatter(x=data[:, 0], y=data[:, 1], color=all_labels,
                    hover_data={'Row': row})

    fig.update_traces(marker=dict(size=6,
                                line=dict(width=2,
                                            )),
                    )
    fig.show()

def s_max(sil_scores):
    s = 0
    for i in sil_scores:
        if i[1] > s:
            s = i[1]
            k = i[0]
    return k

def dimenfix_cluster_scatter_plot(sil_scores, data):
    n_clusters = s_max(sil_scores)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)

    # Plot the points with different colors for each cluster
    plt.scatter(data[:, 0], data[:, 1], s=3 ,c=labels, cmap='viridis')
    plt.title('K-means Clustering')
    plt.show()

def dimenfix_cluster_scatter_plot_2(sil_scores, data, ens_codes, all_labels):
    label_colors = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in ens_codes}
    point_colors = list(map(lambda label: label_colors[label], all_labels))

    n_clusters = s_max(sil_scores)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = point_colors

    # Plot the points with different colors for each cluster
    plt.figure(dpi=100)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=7)
    # plt.title('K-means Clustering')
    legend_labels = list(label_colors.keys())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10) for label in legend_labels]
    plt.legend(legend_handles, legend_labels, title='Origanl Labels', loc = 'upper left', bbox_to_anchor=(1, 1))
    plt.show()

def pca_cumulative_explained_variance(pca_model):
    print("- Percentage of variance explained by each of the selected components:")
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_)*100)
    plt.xlabel("PCA dimension")
    plt.ylabel("Cumulative explained variance %")
    plt.show()
    print("- First three:", pca_model.explained_variance_ratio_[0:3].sum()*100)

def set_labels(ax, reduce_dim_method, dim_x, dim_y):
    ax.set_xlabel(f"{reduce_dim_method} dim {dim_x+1}")
    ax.set_ylabel(f"{reduce_dim_method} dim {dim_y+1}")

def pca_plot_2d_landscapes(ens_codes, reduce_dim_data, reduce_dim_dir, featurization):
    # 2d scatters.
    dim_x = 0
    dim_y = 1
    marker = "."
    legend_kwargs = {"loc": 'upper right',
                    "bbox_to_anchor": (1.1, 1.1),
                    "fontsize": 8}

    # Plot all ensembles at the same time.
    fig, ax = plt.subplots(len(ens_codes)+1, figsize=(4, 4*len(ens_codes)), dpi=120)
    ax[0].set_title("all")
    for code_i in ens_codes:
        ax[0].scatter(reduce_dim_data[code_i][:,dim_x],
                    reduce_dim_data[code_i][:,dim_y],
                    label=code_i, marker=marker)
    ax[0].legend(**legend_kwargs)
    set_labels(ax[0], "pca", dim_x, dim_y)

    # Concatenate all reduced dimensionality data from the dictionary
    all_data = np.concatenate(list(reduce_dim_data.values()))

    # Plot each ensembles.
    for i, code_i in enumerate(ens_codes):
        ax[i+1].set_title(code_i)
        # Plot all data in gray
        ax[i+1].scatter(all_data[:, dim_x],
                        all_data[:, dim_y],
                        label="all", color="gray", alpha=0.25,
                        marker=marker)
        # Plot ensemble data in color
        ax[i+1].scatter(reduce_dim_data[code_i][:,dim_x],
                        reduce_dim_data[code_i][:,dim_y],
                        label=code_i, c=f"C{i}",
                        marker=marker)
        ax[i+1].legend(**legend_kwargs)
        set_labels(ax[i+1], "pca", dim_x, dim_y)

    plt.tight_layout()
    plt.savefig(os.path.join(reduce_dim_dir, 'PCA' + featurization + ens_codes[0]))
    plt.show()

def pca_plot_1d_histograms(ens_codes, concat_reduce_dim_data, reduce_dim_data, reduce_dim_dir, featurization):
    # 1d histograms. Looking at the scatter plot above can be misleading
    # to the eye if we want to assess the density of points. Better use
    # an histogram for a precise evaluation.
    n_bins = 30

    dpi = 120
    fig, ax = plt.subplots(len(ens_codes), 1, figsize=(4, 2*len(ens_codes)), dpi=dpi)
    k = 0
    bins = np.linspace(concat_reduce_dim_data[:,k].min(),
                    concat_reduce_dim_data[:,k].max(),
                    n_bins)

    for i, code_i in enumerate(ens_codes):
        ax[i].hist(reduce_dim_data[code_i][:,k],
                label=code_i,
                bins=bins,
                density=True,
                color=f"C{i}",
                histtype="step")
        ax[i].hist(concat_reduce_dim_data[:,k],
                label="all",
                bins=bins,
                density=True,
                color="gray",
                alpha=0.25,
                histtype="step")
        ax[i].legend(loc='upper right',
                    bbox_to_anchor=(1.1, 1.1),
                    fontsize=8
                    )
        ax[i].set_xlabel(f"Dim {k+1}")
        ax[i].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(os.path.join(reduce_dim_dir, 'PCA_hist' + featurization + ens_codes[0]))
    plt.show()

def pca_correlation_plot(num_residues, sel_dims, feature_names, reduce_dim_model):
    cmap = cm.get_cmap("RdBu")  # RdBu, PiYG
    norm = colors.Normalize(-0.07, 0.07)  # NOTE: this range should be adapted
                                          # when analyzing other systems via PCA!
    dpi = 120

    fig_r = 0.8
    fig, ax = plt.subplots(1, 3, dpi=dpi, figsize=(15*fig_r, 4*fig_r))

    for k, sel_dim in enumerate(sel_dims):
        feature_ids_sorted_by_weight = np.flip(np.argsort(abs(reduce_dim_model.components_[sel_dim,:])))
        matrix = np.zeros((num_residues, num_residues))
        for i in feature_ids_sorted_by_weight:
            r1, r2 = feature_names[i].split("-")
            # Note: this should be patched for proteins with resSeq values not starting from 1!
            matrix[int(r1[3:])-1, int(r2[3:])-1] = reduce_dim_model.components_[sel_dim,i]
            matrix[int(r2[3:])-1, int(r1[3:])-1] = reduce_dim_model.components_[sel_dim,i]
        im = ax[k].imshow(matrix, cmap=cmap, norm=norm)  # RdBu, PiYG
        ax[k].set_xlabel("Residue j")
        ax[k].set_ylabel("Residue i")
        ax[k].set_title(r"Weight of $d_{ij}$" + f" for PCA dim {sel_dim+1}")
        cbar = fig.colorbar(
            im, ax=ax[k],
            label="PCA weight"
        )
    plt.tight_layout()
    plt.show()

def pca_rg_correlation(ens_codes, trajectories, reduce_dim_data, reduce_dim_dir):
    dpi = 120
    fig, ax = plt.subplots(len(ens_codes), 1, figsize=(3, 3*len(ens_codes)), dpi=dpi)
    pca_dim = 0

    for i, code_i in enumerate(ens_codes):
        rg_i = mdtraj.compute_rg(trajectories[code_i])
        ax[i].scatter(reduce_dim_data[code_i][:,pca_dim],
                rg_i, label=code_i,
                color=f"C{i}"
        )
        ax[i].legend(fontsize=8)
        ax[i].set_xlabel(f"Dim {pca_dim+1}")
        ax[i].set_ylabel("Rg [nm]")

    plt.tight_layout()
    plt.savefig(reduce_dim_dir + 'PCA_RG' + ens_codes[0])
    plt.show()

def trajectories_plot_total_sasa(trajectories):
    for ens in trajectories:
        sasa = mdtraj.shrake_rupley(trajectories[ens])


        total_sasa = sasa.sum(axis=1)
        plt.plot(trajectories[ens].time, total_sasa,label = ens )
    plt.xlabel('frame', size=16)
    plt.ylabel('Total SASA (nm)^2', size=16)
    plt.legend()
    plt.show()

def plot_rg_vs_asphericity(trajectories):
    for ens in trajectories:
        x = mdtraj.compute_rg(trajectories[ens])
        y = calculate_asphericity(mdtraj.compute_gyration_tensor(trajectories[ens]))
        
        plt.scatter(x,y,s=4,label = ens)
    plt.ylabel("Asphericity")
    plt.xlabel("Rg [nm]")
    plt.legend()
    plt.show()

def trajectories_plot_density(trajectories):
    for ens in trajectories:
        asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(trajectories[ens]))
        sns.kdeplot(asphericity, label = ens)
    plt.legend()
    plt.show()

def plot_rg_vs_prolateness(trajectories):
    for ens in trajectories:
        x = mdtraj.compute_rg(trajectories[ens])
        y = calculate_prolateness(mdtraj.compute_gyration_tensor(trajectories[ens]))
        
        plt.scatter(x,y,s=4,label = ens)
    plt.ylabel("prolateness")
    plt.xlabel("Rg [nm]")
    plt.legend()
    plt.show()

def trajectories_plot_prolateness(trajectories):
    for ens in trajectories:
        prolatness = calculate_prolateness(mdtraj.compute_gyration_tensor(trajectories[ens]))
        sns.kdeplot(prolatness, label = ens)
    plt.legend()
    plt.show()

def trajectories_plot_dihedrals(trajectories):
    for ens in trajectories:
        four_cons_indices_ca = create_consecutive_indices_matrix(trajectories[ens].topology.select("protein and name CA") )
        ens_dh_ca = mdtraj.compute_dihedrals(trajectories[ens], four_cons_indices_ca).ravel()
        plt.hist(ens_dh_ca, bins=50, histtype="step", density=True, label=ens)
    plt.title("the distribution of dihedral angles between four consecutive Cα beads.")    
    plt.legend()
    plt.show

def get_protein_dssp_data_dict(trajectories):
    dssp_data_dict = {}
    for ens in trajectories:
        dssp_data_dict[ens] = mdtraj.compute_dssp(trajectories[ens])
    return dssp_data_dict

def plot_relative_helix_content(trajectories):
    protein_dssp_data_dict = get_protein_dssp_data_dict(trajectories)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(next(iter(protein_dssp_data_dict.values())).shape[1])

    for protein_name, dssp_data in protein_dssp_data_dict.items():
        # Count the occurrences of 'H' in each column
        h_counts = np.count_nonzero(dssp_data == 'H', axis=0)
        
        # Calculate the total number of residues for each position
        total_residues = dssp_data.shape[0]
        
        # Calculate the relative content of 'H' for each residue
        relative_h_content = h_counts / total_residues
        
        # Plot the relative content for each protein
        ax.bar(range(len(relative_h_content)), relative_h_content, bottom=bottom, label=protein_name)

        bottom += relative_h_content
    
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Relative Content of H (Helix)')
    ax.set_title('Relative Content of H in Each Residue in the ensembles')
    ax.legend()
    plt.show()

def get_rg_data_dict(trajectories):
    rg_dict = {}
    for ens in trajectories:
        #xyz_ens = trajectories[ens].xyz
        rg_dict[ens] = mdtraj.compute_rg(trajectories[ens])
    return rg_dict

def trajectories_plot_rg_comparison(trajectories, n_bins=50, bins_range=(1, 4.5), dpi=96):
    rg_data_dict = get_rg_data_dict(trajectories)
    h_args = {"histtype": "step", "density": True}
    n_systems = len(rg_data_dict)
    bins = np.linspace(bins_range[0], bins_range[1], n_bins + 1)
    fig, ax = plt.subplots(1, n_systems, figsize=(3 * n_systems, 3), dpi=dpi)
    
    for i, (name_i, rg_i) in enumerate(rg_data_dict.items()):
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

def get_distance_matrix_ens_dict(trajectories):
    distance_matrix_ens_dict = {}
    for ens in trajectories:
        xyz_ens = trajectories[ens].xyz[:,trajectories[ens].topology.select("protein and name CA")]
        distance_matrix_ens_dict[ens] = get_distance_matrix(xyz_ens)
    return distance_matrix_ens_dict

def get_contact_ens_dict(trajectories):
    distance_matrix_ens_dict = {}
    contact_ens_dict = {}
    for ens in trajectories:
        xyz_ens = trajectories[ens].xyz[:,trajectories[ens].topology.select("protein and name CA")]
        distance_matrix_ens_dict[ens] = get_distance_matrix(xyz_ens)
        contact_ens_dict[ens] = get_contact_map(distance_matrix_ens_dict[ens])
    return contact_ens_dict

def plot_average_dmap_comparison(trajectories, ticks_fontsize=14,
                                 cbar_fontsize=14,
                                 title_fontsize=14,
                                 dpi=96,
                                 max_d=6.8,
                                 use_ylabel=True):
    ens_dict = get_distance_matrix_ens_dict(trajectories)
    num_proteins = len(ens_dict)
    cols = 2  # Number of columns for subplots
    rows = (num_proteins + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
    axes = axes.reshape((rows, cols))  # Reshape axes to ensure it's 2D
    
    for i, (protein_name, ens_data) in enumerate(ens_dict.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if num_proteins > 1 else axes[0]
        
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

def plot_cmap_comparison(trajectories, title,
                         ticks_fontsize=14,
                         cbar_fontsize=14,
                         title_fontsize=14,
                         dpi=96,
                         cmap_min=-3.5,
                         use_ylabel=True):
    cmap_ens_dict = get_contact_ens_dict(trajectories)
    num_proteins = len(cmap_ens_dict)
    cols = 2  # Number of columns for subplots
    rows = (num_proteins + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
    axes = axes.reshape((rows, cols))  # Reshape axes to ensure it's 2D
    
    cmap = cm.get_cmap("jet")
    norm = colors.Normalize(cmap_min, 0)
    
    for i, (protein_name, cmap_ens) in enumerate(cmap_ens_dict.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if num_proteins > 1 else axes[0]
        
        cmap_ens = np.log10(cmap_ens)
        cmap_ens = np.triu(cmap_ens)
        
        im = ax.imshow(cmap_ens, cmap=cmap, norm=norm)
        ax.set_title(f"Contact Probability Map: {protein_name}", fontsize=title_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
        if not use_ylabel:
            ax.set_yticks([])
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$log_{10}(p_{ij})$', fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        
        im.set_clim(cmap_min, 0)
    
    # Remove any empty subplots
    for i in range(num_proteins, rows * cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=title_fontsize)
    plt.show()

def plot_distance_distribution_multiple(trajectories, dpi=96):
    prot_data_dict = get_distance_matrix_ens_dict(trajectories)
    num_proteins = len(prot_data_dict)
    
    # Set up the subplot grid
    cols = 2  # Number of columns for subplots
    rows = (num_proteins + cols - 1) // cols
    
    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows), dpi=dpi)
    
    # Flatten and plot the distance distribution for each protein
    for i, (protein_name, distance_matrix) in enumerate(prot_data_dict.items()):
        row = i // cols
        col = i % cols
        
        distances_flat = distance_matrix.flatten()
        
        # Plot histogram
        axes_flat = axes.flatten()  # Flatten axes to access them with a single index
        axes_flat[i].hist(distances_flat, bins=50, color='skyblue', edgecolor='black', density=True)
        axes_flat[i].set_title(f'Protein {protein_name}')
        axes_flat[i].set_xlabel('Distance [nm]')
        axes_flat[i].set_ylabel('Density')
        axes_flat[i].grid(True)
    
    # Remove any empty subplots
    for i in range(num_proteins, rows * cols):
        fig.delaxes(axes.flatten()[i])
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def end_to_end_distances_plot(trajectories, atom_selector ="protein and name CA", bins = 50, box_plot = True, means = True, median = True):
    ca_indices = trajectories[next(iter(trajectories))].topology.select(atom_selector)
    dist_list = []
    positions = []
    if box_plot:
        for ens in trajectories:
            positions.append(ens)
            dist_list.append(mdtraj.compute_distances(trajectories[ens],[[ca_indices[0], ca_indices[-1]]]).ravel())
        plt.violinplot(dist_list, showmeans= means, showmedians= median)
        plt.xticks(ticks= [y + 1 for y in range(len(positions))],labels=positions, rotation = 45.0, ha = "center")
        plt.ylabel("End-to-End distance [nm]")
        plt.title("End-to-End distances distribution")
        plt.show()  
    else:
        for ens in trajectories:
            plt.hist(mdtraj.compute_distances(trajectories[ens],[[ca_indices[0], ca_indices[-1]]]).ravel()
                  , label=ens, bins=bins, edgecolor = 'black', density=True)
        plt.title("End-to-End distances distribution")
        plt.legend()
        plt.show()    

def plot_asphericity_dist(trajectories, bins = 50):
    for ens in trajectories:
        asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(trajectories[ens]))
        plt.hist(asphericity, label=ens, bins=bins, edgecolor = 'black', density=True)
    plt.legend()
    plt.show()

def plot_prolateness_dist(trajectories, bins = 50):
    for ens in trajectories:
        prolat = calculate_prolateness(mdtraj.compute_gyration_tensor(trajectories[ens]))
        plt.hist(prolat, label=ens, bins=bins, edgecolor = 'black', density=True)
    plt.legend()
    plt.show()

def plot_alpha_angles_dist(trajectories, bins =50):
    featurizer = FeaturizationFactory.get_featurizer('a_angle')
    for ens in trajectories:
        plt.hist(featurizer.featurize(trajectories[ens]).ravel(), bins=bins, histtype="step", density=True, label=ens)
    plt.title("the distribution of dihedral angles between four consecutive Cα beads.")
    plt.legend()
    plt.show()

def plot_contact_prob(trajectories,title,threshold = 0.8,dpi = 96):

    num_proteins = len(trajectories)
    cols = 2
    rows = (num_proteins + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
    cmap = cm.get_cmap("Blues")
    for i, (protein_name, traj) in enumerate(trajectories.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if num_proteins > 1 else axes

        matrtix_p_map = contact_probability_map(traj , threshold=threshold)
        im = ax.imshow(matrtix_p_map, cmap=cmap )
        ax.set_title(f"Contact Probability Map: {protein_name}", fontsize=14)


        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Frequency', fontsize=14)
        cbar.ax.tick_params(labelsize=14)

    for i in range(num_proteins, rows * cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=14)
    plt.show()
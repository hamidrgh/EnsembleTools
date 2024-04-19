import os
import random
from matplotlib import cm, colors, pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import mdtraj
from matplotlib.lines import Line2D
from dpet.featurization.distances import *
from dpet.featurization.angles import featurize_a_angle
from dpet.visualization.coord import *
from scipy.stats import gaussian_kde

PLOT_DIR = "plots"

def tsne_ramachandran_plot_density(analysis, save=False):
    """
    It gets the ramachadran 2D histogram for the clusters based on the t-SNE
    algorithm when the extracted feature is "phi_psi" 
    """
    
    rama_bins = 50
    rama_linspace = np.linspace(-180, 180, rama_bins)
    
    # Calculate the number of rows and columns for subplots
    num_rows = 1
    num_cols = analysis.reducer.bestK // num_rows if analysis.reducer.bestK % num_rows == 0 else analysis.reducer.bestK // num_rows + 1
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

    # Flatten axes if necessary
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for cluster_id, ax in zip(range(int(analysis.reducer.bestK)), axes):
        cluster_frames = np.where(analysis.reducer.best_kmeans.labels_ == cluster_id)[0]

        phi, psi = np.split(np.degrees(analysis.concat_features[cluster_frames]) , 2 , axis=1)
        phi_flat = phi.flatten()
        psi_flat = psi.flatten()

        hist = ax.hist2d(
        phi_flat,
        psi_flat,
        cmap="viridis",
        bins=(rama_linspace, rama_linspace), 
        norm=colors.LogNorm(),
        density=True)

        ax.set_title(f'Ramachandran Plot for cluster {cluster_id}')
        ax.set_xlabel('Phi (ϕ) Angle (degrees)')
        ax.set_ylabel('Psi (ψ) Angle (degrees)')
    
    key = "tsne_ramachandran_plot_density"
    if key not in analysis.figures:
            analysis.figures[key] = fig

    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(plot_dir  +'/tsnep'+str(int(analysis.reducer.bestP))+'_kmeans'+str(int(analysis.reducer.bestK))+'_ramachandran.png', dpi=800)
    
    plt.tight_layout()
    return fig


def tsne_scatter_plot(analysis, save=False):
    """
    generate the output for t-sne 
    """
    
    bestclust = analysis.reducer.best_kmeans.labels_
    fig , (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14 ,4)) 

    # scatter original  labels
    label_colors = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in analysis.ens_codes}
    point_colors = list(map(lambda label: label_colors[label], analysis.all_labels))
    scatter_labeled = ax1.scatter(analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1], c=point_colors, s=10, alpha = 0.5)
    
    # scatter Rg labels 
    # Rg in Angstrom
    rg_labeled = ax3.scatter(analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1], c= [rg for rg in analysis.rg], s=10, alpha=0.5) 
    cbar = plt.colorbar(rg_labeled, ax=ax3)
    
    # scatter cluster labels
    cmap = cm.get_cmap('jet', analysis.reducer.bestK)
    scatter_cluster = ax2.scatter(analysis.reducer.best_tsne[:,0], analysis.reducer.best_tsne[:,1], c= bestclust.astype(float), s=10,cmap=cmap ,alpha=0.5)
    
    # manage legend
    legend_labels = list(label_colors.keys())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10) for label in legend_labels]
    fig.legend(legend_handles, legend_labels, title='Origanl Labels', loc = 'lower left')

    # KDE plot
    #sns.kdeplot(x=analysis.reducer.best_tsne[:, 0], y=analysis.reducer.best_tsne[:, 1], ax=ax4, fill=True, cmap='Blues', levels=5)
    kde = gaussian_kde([analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1]])
    xi, yi = np.mgrid[min(analysis.reducer.best_tsne[:, 0]):max(analysis.reducer.best_tsne[:, 0]):100j,
                      min(analysis.reducer.best_tsne[:, 1]):max(analysis.reducer.best_tsne[:, 1]):100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
    ax4.contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Blues')
    
    # ax1.scatter(grid_positions[0, densest_indices], grid_positions[1, densest_indices], c='red', marker='x', s=50, label='Densest Points')
    ax1.set_title('Scatter plot (original labels)')
    ax2.set_title('Scatter plot (clustering labels)')
    ax3.set_title('Scatter plot (Rg labels)')
    ax4.set_title('Density Plot ')

    key = "tsne_scatter_plot"
    if key not in analysis.figures:
            analysis.figures[key] = fig
    
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(plot_dir  +'/tsnep'+str(int(analysis.reducer.bestP))+'_kmeans'+str(int(analysis.reducer.bestK))+'_scatter.png', dpi=800)
    return fig

    
def tsne_scatter_plot_rg(analysis, save=False):
    # It plots the same thing as above and could be removed 
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1], c=analysis.rg, cmap='viridis', alpha=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Rg Labels')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('Scatter plot with Rg Labels')

    key = "tsne_scatter_plot_rg"
    if key not in analysis.figures:
            analysis.figures[key] = fig

    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(plot_dir  +'/tsnep'+str(int(analysis.reducer.bestP))+'_kmeans'+str(int(analysis.reducer.bestK))+'_scatter_rg.png', dpi=800)
    return fig
# It was not effiecent way and deleted 
# def s_max(sil_scores):
#     s = 0
#     for i in sil_scores:
#         if i[1] > s:
#             s = i[1]
#             k = i[0]
#     return k

def dimenfix_scatter(analysis, save=False):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,4))

    # scatter original  labels
    label_colors = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in analysis.ens_codes}
    point_colors = list(map(lambda label: label_colors[label], analysis.all_labels))
    scatter_labeled = ax1.scatter(analysis.transformed_data[:,0], analysis.transformed_data[:, 1], c=point_colors, s=10, alpha = 0.5)

    # scatter Rg labels
    rg_labeled = ax3.scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], c= [rg for rg in analysis.rg], s=10, alpha=0.5) 
    cbar = plt.colorbar(rg_labeled, ax=ax3)

    # scatter cluster label
    
    
    kmeans = KMeans(n_clusters= max(analysis.reducer.sil_scores, key=lambda x: x[1])[0], random_state=42)
    labels = kmeans.fit_predict(analysis.transformed_data)
    scatter = ax2.scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], s=10, c=labels, cmap='viridis')

    ax1.set_title('Scatter plot (original labels)')
    ax2.set_title('Scatter plot (clustering labels)')
    ax3.set_title('Scatter plot (Rg labels)')


    #manage legends
    legend_labels = list(label_colors.keys())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10) for label in legend_labels]
    fig.legend(legend_handles, legend_labels, title='Original Labels', loc = 'lower left')

    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(plot_dir  + '/dimenfix_scatter.png', dpi=800)

    return fig

def umap_scatter(analysis, save=False): 
    fig, (ax1, ax3) = plt.subplots(1,2, figsize=(14,4))

    label_colors = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in analysis.ens_codes}
    point_colors = list(map(lambda label: label_colors[label], analysis.all_labels))
    scatter_labeled = ax1.scatter(analysis.transformed_data[:,0], analysis.transformed_data[:, 1], c=point_colors, s=10, alpha = 0.5)

    rg_labeled = ax3.scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], c= [rg for rg in analysis.rg], s=10, alpha=0.5) 
    cbar = plt.colorbar(rg_labeled, ax=ax3)

  

    ax1.set_title('Scatter plot (original labels)')
    ax3.set_title('Scatter plot (Rg labels)')

    legend_labels = list(label_colors.keys())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10) for label in legend_labels]
    fig.legend(legend_handles, legend_labels, title='Original Labels', loc = 'lower left')

    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(plot_dir  + '/umap_scatter.png', dpi=800)

    return fig
    
    


# def dimenfix_scatter_plot_rg(analysis, save=False):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     scatter = ax.scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], c=analysis.rg, cmap='viridis', s=10)
#     fig.colorbar(scatter, ax=ax, label='Rg Numbers')
#     ax.set_xlabel('Dimension 1')
#     ax.set_ylabel('Dimension 2')
#     ax.set_title('Scatter Plot')

#     key = "dimenfix_scatter_plot_rg"
#     if key not in analysis.figures:
#             analysis.figures[key] = fig

#     if save:
#         plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
#         plt.savefig(plot_dir  +'/dimenfix_scatter_rg.png', dpi=800)
#     return fig

# def dimenfix_scatter_plot_ens(analysis, save=False):
#     # Map unique labels to unique integer values
#     label_to_int = {label: i for i, label in enumerate(np.unique(analysis.all_labels))}
    
#     # Convert labels to corresponding integer values
#     int_labels = np.array([label_to_int[label] for label in analysis.all_labels])
    
#     # Create a colormap based on the number of unique labels
#     cmap = plt.cm.get_cmap('viridis', len(label_to_int))
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     scatter = ax.scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], c=int_labels, cmap=cmap, s=100)
#     fig.colorbar(scatter, ax=ax, label='All Labels')
#     ax.set_xlabel('Dimension 1')
#     ax.set_ylabel('Dimension 2')
#     ax.set_title('Scatter Plot 2')

#     key = "dimenfix_scatter_plot_ens"
#     if key not in analysis.figures:
#             analysis.figures[key] = fig

#     if save:
#         plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
#         plt.savefig(plot_dir  +'/dimenfix_scatter_ens.png', dpi=800)
#     return fig



# def dimenfix_cluster_scatter_plot(analysis, save=False):
#     n_clusters = s_max(analysis.reducer.sil_scores)

#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(analysis.transformed_data)

#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Plot the points with different colors for each cluster
#     scatter = ax.scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], s=3, c=labels, cmap='viridis')
#     ax.set_title('K-means Clustering')
    
#     # Create colorbar
#     cbar = fig.colorbar(scatter, ax=ax)
#     cbar.set_label('Cluster Labels')

#     key = "dimenfix_cluster_scatter_plot"
#     if key not in analysis.figures:
#             analysis.figures[key] = fig

#     if save:
#         plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
#         plt.savefig(plot_dir  +'/dimenfix_cluster_scatter.png', dpi=800)
#     return fig

# def dimenfix_cluster_scatter_plot_2(analysis, save=False):
#     label_colors = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in analysis.ens_codes}
#     point_colors = list(map(lambda label: label_colors[label], analysis.all_labels))

#     n_clusters = s_max(analysis.reducer.sil_scores)

#     # Apply K-means clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = point_colors

#     # Create the figure and axis objects
#     fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
#     # Plot the points with different colors for each cluster
#     scatter = ax.scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], c=labels, s=7)
#     ax.set_title('K-means Clustering')
    
#     # Create legend
#     legend_labels = list(label_colors.keys())
#     legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10) for label in legend_labels]
#     ax.legend(legend_handles, legend_labels, title='Original Labels', loc='upper left', bbox_to_anchor=(1, 1))

#     key = "dimenfix_cluster_scatter_plot_2"
#     if key not in analysis.figures:
#             analysis.figures[key] = fig

#     if save:
#         plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
#         plt.savefig(plot_dir  +'/dimenfix_cluster_scatter_2.png', dpi=800)
#     return fig

def pca_cumulative_explained_variance(analysis, save=False):
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(analysis.reduce_dim_model.explained_variance_ratio_) * 100)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Cumulative explained variance %")
    ax.set_title("Cumulative Explained Variance by PCA Dimension")
    ax.grid(True)
    first_three_variance = analysis.reduce_dim_model.explained_variance_ratio_[0:3].sum() * 100
    ax.text(0.5, 0.9, f"First three: {first_three_variance:.2f}%", transform=ax.transAxes, ha='center')
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir, 'PCA_variance' + analysis.featurization + analysis.ens_codes[0]))
    return fig

def set_labels(ax, reduce_dim_method, dim_x, dim_y):
    ax.set_xlabel(f"{reduce_dim_method} dim {dim_x+1}")
    ax.set_ylabel(f"{reduce_dim_method} dim {dim_y+1}")

def pca_plot_2d_landscapes(analysis, save=False):
    # 2d scatters.
    dim_x = 0
    dim_y = 1
    marker = "."
    legend_kwargs = {"loc": 'upper right',
                    "bbox_to_anchor": (1.1, 1.1),
                    "fontsize": 8}

    # Plot all ensembles at the same time.
    fig, ax = plt.subplots(len(analysis.ens_codes)+1, figsize=(4, 4*len(analysis.ens_codes)), dpi=120)
    ax[0].set_title("all")
    for code_i in analysis.ens_codes:
        ax[0].scatter(analysis.reduce_dim_data[code_i][:,dim_x],
                    analysis.reduce_dim_data[code_i][:,dim_y],
                    label=code_i, marker=marker)
    ax[0].legend(**legend_kwargs)
    set_labels(ax[0], "pca", dim_x, dim_y)

    # Concatenate all reduced dimensionality data from the dictionary
    all_data = np.concatenate(list(analysis.reduce_dim_data.values()))

    # Plot each ensembles.
    for i, code_i in enumerate(analysis.ens_codes):
        ax[i+1].set_title(code_i)
        # Plot all data in gray
        ax[i+1].scatter(all_data[:, dim_x],
                        all_data[:, dim_y],
                        label="all", color="gray", alpha=0.25,
                        marker=marker)
        # Plot ensemble data in color
        ax[i+1].scatter(analysis.reduce_dim_data[code_i][:,dim_x],
                        analysis.reduce_dim_data[code_i][:,dim_y],
                        label=code_i, c=f"C{i}",
                        marker=marker)
        ax[i+1].legend(**legend_kwargs)
        set_labels(ax[i+1], "pca", dim_x, dim_y)

    plt.tight_layout()
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir, 'PCA' + analysis.featurization + analysis.ens_codes[0]))
    #plt.show()
    return fig

def pca_plot_1d_histograms(analysis, save=False):
    # 1d histograms. Looking at the scatter plot above can be misleading
    # to the eye if we want to assess the density of points. Better use
    # an histogram for a precise evaluation.
    n_bins = 30

    dpi = 120
    fig, ax = plt.subplots(len(analysis.ens_codes), 1, figsize=(4, 2*len(analysis.ens_codes)), dpi=dpi)
    k = 0
    bins = np.linspace(analysis.transformed_data[:,k].min(),
                    analysis.transformed_data[:,k].max(),
                    n_bins)

    for i, code_i in enumerate(analysis.ens_codes):
        ax[i].hist(analysis.reduce_dim_data[code_i][:,k],
                label=code_i,
                bins=bins,
                density=True,
                color=f"C{i}",
                histtype="step")
        ax[i].hist(analysis.transformed_data[:,k],
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
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir, 'PCA_hist' + analysis.featurization + analysis.ens_codes[0]))
    #plt.show()
    return fig

def pca_correlation_plot(num_residues, sel_dims, analysis, save=False):
    cmap = cm.get_cmap("RdBu")  # RdBu, PiYG
    norm = colors.Normalize(-0.07, 0.07)  # NOTE: this range should be adapted
                                          # when analyzing other systems via PCA!
    dpi = 120

    fig_r = 0.8
    fig, ax = plt.subplots(1, 3, dpi=dpi, figsize=(15*fig_r, 4*fig_r))

    for k, sel_dim in enumerate(sel_dims):
        feature_ids_sorted_by_weight = np.flip(np.argsort(abs(analysis.reduce_dim_model.components_[sel_dim,:])))
        matrix = np.zeros((num_residues, num_residues))
        for i in feature_ids_sorted_by_weight:
            r1, r2 = analysis.feature_names[i].split("-")
            # Note: this should be patched for proteins with resSeq values not starting from 1!
            matrix[int(r1[3:])-1, int(r2[3:])-1] = analysis.reduce_dim_model.components_[sel_dim,i]
            matrix[int(r2[3:])-1, int(r1[3:])-1] = analysis.reduce_dim_model.components_[sel_dim,i]
        im = ax[k].imshow(matrix, cmap=cmap, norm=norm)  # RdBu, PiYG
        ax[k].set_xlabel("Residue j")
        ax[k].set_ylabel("Residue i")
        ax[k].set_title(r"Weight of $d_{ij}$" + f" for PCA dim {sel_dim+1}")
        cbar = fig.colorbar(
            im, ax=ax[k],
            label="PCA weight"
        )
    plt.tight_layout()
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir, 'PCA_correlation' + analysis.featurization + analysis.ens_codes[0]))
    #plt.show()


def pca_rg_correlation(analysis, save=False):
    dpi = 120
    fig, ax = plt.subplots(len(analysis.ens_codes), 1, figsize=(3, 3*len(analysis.ens_codes)), dpi=dpi)
    pca_dim = 0

    for i, code_i in enumerate(analysis.ens_codes):
        rg_i = mdtraj.compute_rg(analysis.trajectories[code_i])
        ax[i].scatter(analysis.reduce_dim_data[code_i][:,pca_dim],
                rg_i, label=code_i,
                color=f"C{i}"
        )
        ax[i].legend(fontsize=8)
        ax[i].set_xlabel(f"Dim {pca_dim+1}")
        ax[i].set_ylabel("Rg [nm]")

    plt.tight_layout()
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir,'PCA_RG' + analysis.ens_codes[0]))
    #plt.show()
    return fig

def plot_global_sasa(analysis,showmeans=True, showmedians=True ,save=False):
    fig, ax = plt.subplots(1, 1 )
    positions = []
    dist_list = []
    for ens in analysis.trajectories:
        positions.append(ens)
        sasa = mdtraj.shrake_rupley(analysis.trajectories[ens])
        total_sasa = sasa.sum(axis=1)
        dist_list.append(total_sasa)
    ax.violinplot(dist_list, showmeans=showmeans, showmedians=showmedians  )

    plt.xticks(ticks= [y + 1 for y in range(len(positions))],labels=positions, rotation = 45.0, ha = "center")
    plt.title('SASA distribution over the ensembles')
    plt.ylabel('SASA (nm)^2')
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir,'Global_SASA_dist' + analysis.ens_codes[0]))
    # plt.legend()
    # plt.show()

def plot_rg_vs_asphericity(analysis, save=False):
    for ens in analysis.trajectories:
        x = mdtraj.compute_rg(analysis.trajectories[ens])
        y = calculate_asphericity(mdtraj.compute_gyration_tensor(analysis.trajectories[ens]))
        p = np.corrcoef(x , y)
        plt.scatter(x,y,s=4,label = ens)
        print(f"Pearson coeff for {ens} = {round(p[0][1], 3)}")
    plt.ylabel("Asphericity")
    plt.xlabel("Rg [nm]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir,'Rg_vs_Asphericity' + analysis.ens_codes[0]))
    # plt.show()

'''
def trajectories_plot_density(trajectories):
    for ens in trajectories:
        asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(trajectories[ens]))
        sns.kdeplot(asphericity, label = ens)
    plt.legend()
    plt.show()
'''
'''
def trajectories_plot_density(trajectories):
    fig, ax = plt.subplots()
    for ens in trajectories:
        asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(trajectories[ens]))
        # sns.kdeplot(asphericity, label = ens)
        kde = gaussian_kde(asphericity)
        x = np.linspace(min(asphericity), max(asphericity), 100)
        ax.plot(x, kde(x), label=ens)
    ax.legend()
    ax.set_xlabel('Asphericity')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimate of Asphericity for Trajectories')
    plt.show()
'''
def plot_rg_vs_prolateness(analysis, save=False):
    for ens in analysis.trajectories:
        x = mdtraj.compute_rg(analysis.trajectories[ens])
        y = calculate_prolateness(mdtraj.compute_gyration_tensor(analysis.trajectories[ens]))
        p = np.corrcoef(x , y)
        plt.scatter(x,y,s=4,label = ens)
        print(f"Pearson coeff for {ens} = {round(p[0][1], 3)}")
    plt.ylabel("prolateness")
    plt.xlabel("Rg [nm]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir,'Rg_vs_Prolateness' + analysis.ens_codes[0]))
    # plt.show()

'''
def trajectories_plot_prolateness(trajectories):
    for ens in trajectories:
        prolatness = calculate_prolateness(mdtraj.compute_gyration_tensor(trajectories[ens]))
        sns.kdeplot(prolatness, label = ens)
    plt.legend()
    plt.show()
'''
'''
def trajectories_plot_prolateness(trajectories):
    fig, ax = plt.subplots()
    for ens in trajectories:
        prolatness = calculate_prolateness(mdtraj.compute_gyration_tensor(trajectories[ens]))
        # sns.kdeplot(asphericity, label = ens)
        kde = gaussian_kde(prolatness)
        x = np.linspace(min(prolatness), max(prolatness), 100)
        ax.plot(x, kde(x), label=ens)
    ax.legend()
    ax.set_xlabel('Prolateness')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimate of Prolateness for Trajectories')
    plt.show()
'''

def plot_alpha_angle_dihederal(analysis, bins=50, atom_selector='protein and name CA', save=False):
    for ens in analysis.trajectories:
        four_cons_indices_ca = create_consecutive_indices_matrix(analysis.trajectories[ens].topology.select(atom_selector) )
        ens_dh_ca = mdtraj.compute_dihedrals(analysis.trajectories[ens], four_cons_indices_ca).ravel()
        plt.hist(ens_dh_ca, bins=bins, histtype="step", density=True, label=ens)
    plt.title("the distribution of dihedral angles between four consecutive Cα beads.")    
    plt.legend()
    if save:
        plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
        plt.savefig(os.path.join(plot_dir,'alpha_angle_dihederal' + analysis.ens_codes[0]))
    # plt.show

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
        ax.plot(range(len(relative_h_content)), relative_h_content,marker='o', linestyle='dashed' ,label=protein_name, alpha= 0.5)

        bottom += relative_h_content
    
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Relative Content of H (Helix)')
    ax.set_title('Relative Content of H in Each Residue in the ensembles')
    ax.legend(bbox_to_anchor=(1.04,0), loc = "lower left")
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
        if n_systems > 1:
            ax[i].hist(rg_i, bins=bins, label=name_i, **h_args)
            ax[i].set_title(name_i)
            if i == 0:
                ax[i].set_ylabel("Density")
            ax[i].set_xlabel("Rg [nm]")
            mean_rg = np.mean(rg_i)
            median_rg = np.median(rg_i)

            mean_line = ax[i].axvline(mean_rg, color='k', linestyle='dashed', linewidth=1)
            median_line = ax[i].axvline(median_rg, color='r', linestyle='dashed', linewidth=1)
        elif n_systems == 1:
            ax.hist(rg_i, bins=bins, label=name_i, **h_args)
            ax.set_title(name_i)
            if i == 0:
                ax.set_ylabel("Density")
            ax.set_xlabel("Rg [nm]")
            mean_rg = np.mean(rg_i)
            median_rg = np.median(rg_i)

            mean_line = ax.axvline(mean_rg, color='k', linestyle='dashed', linewidth=1)
            median_line = ax.axvline(median_rg, color='r', linestyle='dashed', linewidth=1)

    
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
    
    for ax, (protein_name, ens_data) in zip(fig.axes, (ens_dict.items())):
       
        ax = ax
        
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

def end_to_end_distances_plot(trajectories, atom_selector ="name CA", bins = 50, violin_plot = True, means = True, median = True):
    dist_list = []
    positions = []
    if violin_plot:
        for ens in trajectories:
            ca_indices = trajectories[ens].topology.select(atom_selector)
            positions.append(ens)
            dist_list.append(mdtraj.compute_distances(trajectories[ens],[[ca_indices[0], ca_indices[-1]]]).ravel())
        plt.violinplot(dist_list, showmeans= means, showmedians= median)
        plt.xticks(ticks= [y + 1 for y in range(len(positions))],labels=positions, rotation = 45.0, ha = "center")
        plt.ylabel("End-to-End distance [nm]")
        plt.title("End-to-End distances distribution")
        plt.show()  
    else:
        for ens in trajectories:
            ca_indices = trajectories[ens].topology.select(atom_selector)
            plt.hist(mdtraj.compute_distances(trajectories[ens],[[ca_indices[0], ca_indices[-1]]]).ravel()
                  , label=ens, bins=bins, edgecolor = 'black', density=True)
        plt.title("End-to-End distances distribution")
        plt.legend()
        plt.show()    

def plot_asphericity_dist(trajectories, bins = 50, violin_plot = True, means = True, median = True ):
    asph_list = []
    positions = []
    if violin_plot:
        for ens in trajectories:
            asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(trajectories[ens]))
            asph_list.append(asphericity)
            positions.append(ens)
        plt.violinplot(asph_list, showmeans=means, showmedians= median)
        plt.xticks(ticks= [y +1 for y in range(len(positions))],labels=positions, rotation = 45.0, ha = "center")
        plt.ylabel("Asphericity")
        plt.title("Asphericity distribution")
        plt.show()
    else:
        for ens in trajectories:
            asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(trajectories[ens]))
            plt.hist(asphericity, label=ens, bins=bins, edgecolor = 'black', density=True)
        plt.title("Asphericity distribution")        
        plt.legend()
        plt.show()

def plot_prolateness_dist(trajectories, bins= 50, violin_plot= True, median=False, mean=False  ):
    prolat_list = []
    positions = []
    if violin_plot:
        for ens in trajectories:
            prolat = calculate_prolateness(mdtraj.compute_gyration_tensor(trajectories[ens]))
            prolat_list.append(prolat)
            positions.append(ens)
        plt.violinplot(prolat_list, showmeans= mean, showmedians= median)
        plt.xticks(ticks= [y +1 for y in range(len(positions))],labels=positions, rotation = 45.0, ha = "center")
        plt.ylabel("Prolateness")
        plt.title("Prolateness distribution")
        plt.show()
    else:
        for ens in trajectories:
            plt.hist(prolat, label=ens, bins=bins, edgecolor = 'black', density=True)
        plt.title("Prolateness distribution")
        plt.legend()
        plt.show()

def plot_alpha_angles_dist(trajectories, bins =50):
    for ens in trajectories:
        plt.hist(featurize_a_angle(trajectories[ens], get_names=False).ravel(), bins=bins, histtype="step", density=False, label=ens)
        
    plt.title("the distribution of dihedral angles between four consecutive Cα beads.")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_contact_prob(trajectories,threshold = 0.8,dpi = 96):


    num_proteins = len(trajectories)
    
    cols = 2
    rows = (num_proteins + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
    cmap = cm.get_cmap("Blues")
    axes = axes.reshape((rows, cols))  # Reshape axes to ensure it's 2D
    for (protein_name, traj) , ax in zip((trajectories.items()), fig.axes):
        
        matrtix_p_map = contact_probability_map(traj , threshold=threshold)
        im = ax.imshow(matrtix_p_map, cmap=cmap )
        ax.set_title(f"Contact Probability Map: {protein_name}", fontsize=14)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Frequency', fontsize=14)
        cbar.ax.tick_params(labelsize=14)

        # Remove any empty subplots
    for i in range(num_proteins, rows * cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

def plot_ramachandran_plot(trajectories, two_d_hist= True, linespaces = (-180, 180, 80)):
    if two_d_hist:
        fig, axes = plt.subplots(1, len(trajectories), figsize=(5*len(trajectories), 5))
        rama_linspace = np.linspace(linespaces[0], linespaces[1], linespaces[2])
        for ens, ax in zip(trajectories, axes.ravel()):
            phi_flat = np.degrees(mdtraj.compute_phi(trajectories[ens])[1]).ravel()
            psi_flat = np.degrees(mdtraj.compute_psi(trajectories[ens])[1]).ravel()
            hist = ax.hist2d(
            phi_flat,
            psi_flat,
            cmap="viridis",
            bins=(rama_linspace, rama_linspace), 
            norm=colors.LogNorm(),
            density=True)

            ax.set_title(f'Ramachandran Plot for cluster {ens}')
            ax.set_xlabel('Phi (ϕ) Angle (degrees)')
            ax.set_ylabel('Psi (ψ) Angle (degrees)')

        plt.tight_layout()
        plt.show()
    else:
        fig,ax = plt.subplots(1,1)
        for ens in trajectories:
            phi = np.degrees(mdtraj.compute_phi(trajectories[ens])[1])
            psi = np.degrees(mdtraj.compute_psi(trajectories[ens])[1])
            plt.scatter(phi, psi, s=1, label= ens)
        ax.set_xlabel('Phi (ϕ) Angle (degrees)')
        ax.set_ylabel('Psi (ψ) Angle (degrees)')
        plt.legend(bbox_to_anchor=(1.04,0), loc = "lower left")
        plt.show()

def plot_ss_measure_disorder( featurized_data: dict, pointer: list = None, figsize=(15,5)):
    f = ss_measure_disorder(featurized_data)
    fig, axes = plt.subplots(1,1, figsize=figsize)
    keys = list(featurized_data.keys())
    x = [i+1 for i in range(len(f[keys[0]]))]
    for key, values in f.items():
        axes.scatter(x, values, label= key)
    
    axes.set_xticks([i for i in x if  i==1 or i%5 == 0])
    axes.set_xlabel("Residue Index")
    axes.set_ylabel("Site-specific measure of disorder")
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if pointer is not None:
        for res in pointer:
            axes.axvline(x= res, c= 'blue', linestyle= '--', alpha= 0.3, linewidth= 1)
    plt.show()
        
def plot_ss_order_parameter(trajectories, pointer:list= None , figsize=(15,5) ): 
    dict_ca_xyz = {}
    for ens in trajectories:
        ca_index= trajectories[ens].topology.select('name == CA')
        dict_ca_xyz[ens] = trajectories[ens].xyz[:,ca_index,:]

    dict_order_parameter = site_specific_order_parameter(dict_ca_xyz)
    fig, axes = plt.subplots(1,1, figsize=figsize)
    keys = list(dict_order_parameter.keys())
    x = [i+1 for i in range(len(dict_order_parameter[keys[0]]))]
    for key, values in dict_order_parameter.items():
        axes.scatter(x, values, label= key, alpha= 0.5)
    axes.set_xticks([i for i in x if  i==1 or i%5 == 0])
    axes.set_xlabel("Residue Index")
    axes.set_ylabel("Site-specific order parameter")
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if pointer is not None:
        for res in pointer:
            axes.axvline(x=res, c='blue', linestyle='--', alpha=0.3, linewidth=1)
        
    plt.show()

def plot_local_sasa(analysis, figsize=(15,5), pointer:list= None): 
    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, ens in enumerate(analysis.trajectories):

        res_based_sasa = mdtraj.shrake_rupley(analysis.trajectories[ens], mode='residue')
        sasa_mean = np.mean(res_based_sasa, axis=0)
        sasa_std = np.std(res_based_sasa, axis=0)        

        ax.plot(np.arange(1, len(sasa_mean)+1), sasa_mean, '-o', color=colors[i % len(colors)], label= ens)
        ax.fill_between(np.arange(1,len(sasa_mean)+1), sasa_mean - sasa_std, sasa_mean + sasa_std, alpha=0.3, color=colors[i % len(colors)])

    ax.set_xticks([i for i in np.arange(1,len(sasa_mean)+1) if  i==1 or i%5 == 0])
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Mean SASA')
    ax.set_title('Mean SASA for Each Residue in Ensembles')
    ax.legend()
    ax.grid(True)
    if pointer is not None:
        for res in pointer:
            ax.axvline(x=res, c='blue', linestyle='--', alpha=0.3, linewidth=1)
        

    plt.tight_layout()
    plt.show()

def plot_dist_ca_com(analysis, min_sep=2,max_sep=None ,get_names=True,inverse=False ,figsize=(6,2.5)):

    for ens in analysis.trajectories:
        traj = analysis.trajectories[ens]
        feat, names = featurize_com_dist(traj=traj, min_sep=min_sep,max_sep=max_sep,inverse=inverse ,get_names=get_names)  # Compute (N, *) feature arrays.
        print(f"# Ensemble: {ens}")
        print("features:", feat.shape)

        com_dmap = calc_ca_dmap(traj=traj)
        com_dmap_mean = com_dmap.mean(axis=0)
        ca_dmap = calc_ca_dmap(traj=traj)
        ca_dmap_mean = ca_dmap.mean(axis=0)

        print("distance matrix:", com_dmap_mean.shape)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(ens)
        im0 = ax[0].imshow(ca_dmap_mean)
        ax[0].set_title("CA")
        im1 = ax[1].imshow(com_dmap_mean)
        ax[1].set_title("COM")
        cbar = fig.colorbar(im0, ax=ax[0], shrink=0.8)
        cbar.set_label("distance [nm]")
        cbar = fig.colorbar(im1, ax=ax[1], shrink=0.8)
        cbar.set_label("distance [nm]")

        plt.tight_layout()
        plt.show()


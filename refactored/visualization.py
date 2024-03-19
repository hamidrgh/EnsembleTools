import os
import random
from matplotlib import cm, colors, pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import plotly.graph_objects as go 
import plotly.express as px
import mdtraj

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
    plt.savefig(os.path.join(reduce_dim_dir, 'PCA' + featurization + ens_codes[0][0] + ens_codes[0][1]))
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
    plt.savefig(os.path.join(reduce_dim_dir, 'PCA_hist' + featurization + ens_codes[0][0] + ens_codes[0][1]))
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
    plt.savefig(reduce_dim_dir + 'PCA_RG' + ens_codes[0][0] + ens_codes[0][1])
    plt.show()
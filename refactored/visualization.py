import random
from matplotlib import cm, pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

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
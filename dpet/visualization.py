import os
import random
from typing import List, Tuple
from matplotlib import cm, colors, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.cluster import KMeans
import mdtraj
from matplotlib.lines import Line2D
from dpet.featurization.distances import *
from dpet.ensemble_analysis import EnsembleAnalysis
from dpet.featurization.angles import featurize_a_angle
from dpet.data.coord import *
from scipy.stats import gaussian_kde

PLOT_DIR = "plots"

class Visualization:

    def __init__(self, analysis: EnsembleAnalysis):
        self.analysis = analysis
        self.plot_dir = os.path.join(self.analysis.data_dir, PLOT_DIR)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.figures = {}

    def tsne_ramachandran_plot_density(self, save:bool=False):

        """
        Plot the 2-D histogram ramachandran plots of the
        clusters from t-SNE analysis. \n
        The results is only meaningful when the extracted feature is "phi_psi".

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method != "tsne" or analysis.featurization != "phi_psi":
                print("This analysis is only valid for t-SNE reduction with phi_psi feature extraction.")
                return
        
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

        plt.tight_layout()

        self.figures["tsne_ramachandran_plot_density"] = fig

        if save:
            plt.savefig(self.plot_dir  +'/tsnep'+str(int(analysis.reducer.bestP))+'_kmeans'+str(int(analysis.reducer.bestK))+'_ramachandran.png', dpi=800)

        return axes

    def tsne_scatter_plot(self, save:bool=False):

        """
        Plot the results of t-SNE analysis. 
        Three scatter plot will be generated based on original, clustering and Rg labels. 
        One KDE density plot will also be generated to show the most populated areas in 
        the reduced dimension.   

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method != "tsne":
                print("Analysis is only valid for t-SNE dimensionality reduction.")
                return
        
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

        self.figures["tsne_scatter_plot"] = fig
        
        if save:
            plt.savefig(self.plot_dir  +'/tsnep'+str(int(analysis.reducer.bestP))+'_kmeans'+str(int(analysis.reducer.bestK))+'_scatter.png', dpi=800)

        return ax1, ax2, ax3, ax4

    def dimenfix_scatter(self, save:bool=False):

        """
        Plot the the complete results for dimenfix method. 

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method != "dimenfix":
                print("Analysis is only valid for dimenfix dimensionality reduction.")
                return
        
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
        fig.legend(legend_handles, legend_labels, title='Origanl Labels', loc = 'lower left')

        self.figures["dimenfix_scatter"] = fig

        if save:
            plt.savefig(self.plot_dir  + '/dimenfix_scatter.png', dpi=800)
        
        return ax1, ax2, ax3
        
    def umap_scatter(self, save:bool=False): 

        """
        Generate a scatter plot of the transformed data using UMAP.

        Parameters
        ----------
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        Notes
        -----
        This method creates a scatter plot of the transformed data using UMAP, with two subplots:
        - The first subplot shows the scatter plot with points colored by original labels.
        - The second subplot shows the scatter plot with points colored by Rg labels.

        """

        analysis = self.analysis
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

        self.figures["umap_scatter"] = fig

        if save:
            plot_dir = os.path.join(analysis.data_dir, PLOT_DIR)
            plt.savefig(plot_dir  + '/umap_scatter.png', dpi=800)

        return ax1, ax3


    def pca_cumulative_explained_variance(self, save:bool=False):

        """
        Plot the cumulative variance. Only applicable when the
        dimensionality reduction method is "pca"

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        
        analysis = self.analysis

        if analysis.reduce_dim_method != "pca":
                print("Analysis is only valid for pca dimensionality reduction.")
                return
        
        fig, ax = plt.subplots()
        ax.plot(np.cumsum(analysis.reduce_dim_model.explained_variance_ratio_) * 100)
        ax.set_xlabel("PCA dimension")
        ax.set_ylabel("Cumulative explained variance %")
        ax.set_title("Cumulative Explained Variance by PCA Dimension")
        ax.grid(True)
        first_three_variance = analysis.reduce_dim_model.explained_variance_ratio_[0:3].sum() * 100
        ax.text(0.5, 0.9, f"First three: {first_three_variance:.2f}%", transform=ax.transAxes, ha='center')

        self.figures["pca_cumulative_explained_variance"] = fig

        if save:
            plt.savefig(os.path.join(self.plot_dir, 'PCA_variance' + analysis.featurization + analysis.ens_codes[0]))

        return ax

    def _set_labels(self, ax, reduce_dim_method, dim_x, dim_y):
        ax.set_xlabel(f"{reduce_dim_method} dim {dim_x+1}")
        ax.set_ylabel(f"{reduce_dim_method} dim {dim_y+1}")

    def pca_plot_2d_landscapes(self, save:bool=False):

        """
        Plot 2D landscapes when the dimensionality reduction method 
        is "pca" or "kpca"

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        
        analysis = self.analysis

        if analysis.reduce_dim_method not in ("pca","kpca"):
                print("Analysis is only valid for pca dimensionality reduction.")
                return

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
        for ens_code, ensemble in analysis.ensembles.items():
            ax[0].scatter(ensemble.reduce_dim_data[:,dim_x],
                        ensemble.reduce_dim_data[:,dim_y],
                        label=ens_code, marker=marker)
        ax[0].legend(**legend_kwargs)
        self._set_labels(ax[0], "pca", dim_x, dim_y)

        # Concatenate all reduced dimensionality data from the dictionary
        all_data = analysis.transformed_data

        # Plot each ensembles.
        for i, (ens_code, ensemble) in enumerate(analysis.ensembles.items()):
            ax[i+1].set_title(ens_code)
            # Plot all data in gray
            ax[i+1].scatter(all_data[:, dim_x],
                            all_data[:, dim_y],
                            label="all", color="gray", alpha=0.25,
                            marker=marker)
            # Plot ensemble data in color
            ax[i+1].scatter(ensemble.reduce_dim_data[:,dim_x],
                            ensemble.reduce_dim_data[:,dim_y],
                            label=ens_code, c=f"C{i}",
                            marker=marker)
            ax[i+1].legend(**legend_kwargs)
            self._set_labels(ax[i+1], "pca", dim_x, dim_y)

        plt.tight_layout()

        self.figures["pca_plot_2d_landscapes"] = fig
        if save:
            plt.savefig(os.path.join(self.plot_dir, 'PCA_2d_landscapes_' + analysis.featurization + analysis.ens_codes[0]))

        return ax

    def pca_plot_1d_histograms(self, save:bool=False):
        
        """
        Plot 1D histogram when the dimensionality reduction method 
        is "pca" or "kpca"

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method not in ("pca", "kpca"):
                print("Analysis is only valid for pca and kpca dimensionality reduction.")
                return
        
        n_bins = 30

        dpi = 120
        fig, ax = plt.subplots(len(analysis.ens_codes), 1, figsize=(4, 2*len(analysis.ens_codes)), dpi=dpi)
        k = 0
        bins = np.linspace(analysis.transformed_data[:,k].min(),
                        analysis.transformed_data[:,k].max(),
                        n_bins)
        
        # Ensure ax is always a list
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        for i, (ens_code, ensemble) in enumerate(analysis.ensembles.items()):
            ax[i].hist(ensemble.reduce_dim_data[:,k],
                    label=ens_code,
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
        self.figures["pca_plot_1d_histograms"] = fig
        if save:
            plt.savefig(os.path.join(self.plot_dir, 'PCA_hist' + analysis.featurization + analysis.ens_codes[0]))
        
        return ax


    def pca_correlation_plot(self, num_residues: int, sel_dims: List[int], save: bool = False):
        """
        Plot the correlation between residues based on PCA weights.

        Parameters
        ----------
        num_residues : int
            The total number of residues in the system.
        sel_dims : List[int]
            A list of indices specifying the PCA dimensions to include in the plot.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        Notes
        -----
        This method generates a correlation plot showing the weights of pairwise residue distances
        for selected PCA dimensions. The plot visualizes the correlation between residues based on
        the PCA weights.

        The analysis is only valid on PCA and kernel PCA dimensionality reduction with 'ca_dist' feature extraction.

        """
        analysis = self.analysis

        if analysis.reduce_dim_method not in ("pca", "kpca") or analysis.featurization != "ca_dist":
                print("Analysis is only valid for pca and kpca dimensionality reduction with ca_dist feature extraction.")
                return
        
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
        self.figures["pca_correlation_plot"] = fig
        if save:
            plt.savefig(os.path.join(self.plot_dir, 'PCA_correlation' + analysis.featurization + analysis.ens_codes[0]))

        return ax


    def pca_rg_correlation(self, save: bool = False):

        """
        Examine and plot the correlation between PC dimension 1 and the amount of Rg
        Typically high correlation can be detected here. 

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        
        analysis = self.analysis

        if analysis.reduce_dim_method not in ("pca", "kpca"):
                print("Analysis is only valid for pca and kpca dimensionality reduction.")
                return
        
        dpi = 120
        fig, ax = plt.subplots(len(analysis.ens_codes), 1, figsize=(3, 3*len(analysis.ens_codes)), dpi=dpi)
        pca_dim = 0

        # Ensure ax is always a list
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        for i, (ens_code, ensemble) in enumerate(analysis.ensembles.items()):
            rg_i = mdtraj.compute_rg(ensemble.trajectory)
            ax[i].scatter(ensemble.reduce_dim_data[:,pca_dim],
                    rg_i, label=ens_code,
                    color=f"C{i}"
            )
            ax[i].legend(fontsize=8)
            ax[i].set_xlabel(f"Dim {pca_dim+1}")
            ax[i].set_ylabel("Rg [nm]")

        plt.tight_layout()
        self.figures["pca_rg_correlation"] = fig
        if save:
            plt.savefig(os.path.join(self.plot_dir,'PCA_RG' + analysis.ens_codes[0]))

        return ax

    def plot_global_sasa(self, showmeans: bool = True, showmedians: bool = True ,save: bool =False):

        """
        Plot the distribution of SASA for each conformation 
        within the ensembles.

        Parameters
        ----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.

        showmean: bool, optional
            if True it will show the mean. Default is True.

        showmedian: bool, optional
            if True it will show the median. Default is True.
        
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """

        analysis = self.analysis

        fig, ax = plt.subplots(1, 1 )
        positions = []
        dist_list = []
        for ens in analysis.ensembles:
            positions.append(ens)
            sasa = mdtraj.shrake_rupley(analysis.ensembles[ens].trajectory)
            total_sasa = sasa.sum(axis=1)
            dist_list.append(total_sasa)
        ax.violinplot(dist_list, showmeans=showmeans, showmedians=showmedians  )

        ax.set_xticks(ticks= [y + 1 for y in range(len(positions))])
        ax.set_xticklabels(positions, rotation = 45.0, ha = "center")
        ax.set_title('SASA distribution over the ensembles')
        ax.set_ylabel('SASA (nm)^2')
        self.figures["plot_global_sasa"] = fig
        if save:
            plt.savefig(os.path.join(self.plot_dir,'Global_SASA_dist' + analysis.ens_codes[0]))
        return ax

    def plot_rg_vs_asphericity(self, save: bool = False):
        """
        Plot the Rg versus Asphericity and get the pearson correlation coefficient to evaluate 
        the correlation between Rg and Asphericity.
        
        Parameters:
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        
        analysis = self.analysis
        
        fig, ax = plt.subplots()  # Create a new figure
        
        for ens_code, ensemble in analysis.ensembles.items():
            x = mdtraj.compute_rg(ensemble.trajectory)
            y = calculate_asphericity(mdtraj.compute_gyration_tensor(ensemble.trajectory))
            p = np.corrcoef(x, y)
            ax.scatter(x, y, s=4, label=ens_code)
            print(f"Pearson coeff for {ens_code} = {round(p[0][1], 3)}")
        
        ax.set_ylabel("Asphericity")
        ax.set_xlabel("Rg [nm]")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.figures["plot_rg_vs_asphericity"] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'Rg_vs_Asphericity' + analysis.ens_codes[0]))
        
        return ax
    
  
    def plot_rg_vs_prolateness(self, save: bool = False):
        """
        Plot the Rg versus Prolateness and get the Pearson correlation coefficient to evaluate 
        the correlation between Rg and Prolateness. 

        Parameters
        ----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.

        """
        analysis = self.analysis
        
        # Create a new figure object
        fig, ax = plt.subplots()

        for ens_code, ensemble in analysis.ensembles.items():
            x = mdtraj.compute_rg(ensemble.trajectory)
            y = calculate_prolateness(mdtraj.compute_gyration_tensor(ensemble.trajectory))
            p = np.corrcoef(x, y)
            ax.scatter(x, y, s=4, label=ens_code)
            print(f"Pearson coeff for {ens_code} = {round(p[0][1], 3)}")

        ax.set_ylabel("Prolateness")
        ax.set_xlabel("Rg [nm]")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        self.figures["plot_rg_vs_prolateness"] = fig

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'Rg_vs_Prolateness' + analysis.ens_codes[0]))
        
        return ax


    def plot_alpha_angle_dihedral(self, bins: int = 50, save: bool = False):
        """
        Plot the distribution of dihedral angles between four consecutive Cα beads.

        Parameters
        ----------
        bins : int, optional
            Number of bins for the histogram. Default is 50.
        save : bool, optional
            If True, the plot will be saved in the data directory. Default is False.

        """
        analysis = self.analysis
        
        # Create a new figure object
        fig, ax = plt.subplots()

        for ens_code, ensemble in analysis.ensembles.items():
            four_cons_indices_ca = create_consecutive_indices_matrix(ensemble.trajectory.topology.select(ensemble.atom_selector) )
            ens_dh_ca = mdtraj.compute_dihedrals(ensemble.trajectory, four_cons_indices_ca).ravel()
            ax.hist(ens_dh_ca, bins=bins, histtype="step", density=True, label=ens_code)

        ax.set_title("The distribution of dihedral angles between four consecutive Cα beads.")
        ax.legend()

        self.figures["plot_alpha_angle_dihedral"] = fig

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'alpha_angle_dihedral' + analysis.ens_codes[0]))
        
        return ax


    def _get_protein_dssp_data_dict(self):
        ensembles = self.analysis.ensembles
        dssp_data_dict = {}
        for ens_code, ensemble in ensembles.items():
            dssp_data_dict[ens_code] = mdtraj.compute_dssp(ensemble.trajectory)
        return dssp_data_dict

    def plot_relative_helix_content(self, save: bool = False):
        """
        Plot the relative helix content in each ensemble for each residue. 

        Parameters
        ----------
        save : bool, optional
            If True, the plot will be saved in the data directory. Default is False.

        """
        if self.analysis.exists_coarse_grained():
            print("This analysis is not possible with coarse-grained models.")
            return
        protein_dssp_data_dict = self._get_protein_dssp_data_dict()
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

        self.figures["plot_relative_helix_content"] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'relative_helix_' + self.analysis.ens_codes[0]))
        
        return ax

    def _get_rg_data_dict(self):
        ensembles = self.analysis.ensembles
        rg_dict = {}
        for ens_code, ensemble in ensembles.items():
            rg_dict[ens_code] = mdtraj.compute_rg(ensemble.trajectory)
        return rg_dict

    def trajectories_plot_rg_comparison(self, n_bins: int = 50, dpi: int = 96, save: bool = False):
        """
        Plot the distribution of the radius of gyration (Rg) within each ensemble.

        Parameters
        ----------
        n_bins : int, optional
            The number of bins for the histogram. Default is 50.
        dpi : int, optional
            The DPI (dots per inch) of the output figure. Default is 96.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.


        Notes
        -----
        This method plots the distribution of the radius of gyration (Rg) within each ensemble in the analysis.

        The Rg values are binned according to the specified number of bins (`n_bins`) and range (`bins_range`) and 
        displayed as histograms. Additionally, dashed lines representing the mean and median Rg values are overlaid
        on each histogram.

        """

        rg_data_dict = self._get_rg_data_dict()
        concatenated_rg_data = np.concatenate(list(rg_data_dict.values())) 
        min_value = np.floor(np.min(concatenated_rg_data)) #calculating min and max Rg to automate the range
        max_value = np.ceil(np.max(concatenated_rg_data))
        h_args = {"histtype": "step", "density": True}
        n_systems = len(rg_data_dict)
        
        bins = np.linspace(min_value, max_value, n_bins + 1)
        fig, ax = plt.subplots(1, n_systems, figsize=(3 * n_systems, 3), dpi=dpi)
        
        # Ensure ax is always a list
        if not isinstance(ax, np.ndarray):
            ax = [ax]

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

        self.figures['trajectories_plot_rg_comparison'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'rg_comparison_' + self.analysis.ens_codes[0]))

        return ax


    def _get_distance_matrix_ens_dict(self):
        ensembles = self.analysis.ensembles
        distance_matrix_ens_dict = {}
        for ens_code, ensemble in ensembles.items():
            selector = ensemble.atom_selector
            trajectory = ensemble.trajectory
            xyz_ens = trajectory.xyz[:,trajectory.topology.select(selector)]
            distance_matrix_ens_dict[ens_code] = get_distance_matrix(xyz_ens)
        return distance_matrix_ens_dict

    def _get_contact_ens_dict(self):
        ensembles = self.analysis.ensembles
        distance_matrix_ens_dict = {}
        contact_ens_dict = {}
        for ens_code, ensemble in ensembles.items():
            xyz_ens = ensemble.trajectory.xyz[:,ensemble.trajectory.topology.select(ensemble.atom_selector)]
            distance_matrix_ens_dict[ens_code] = get_distance_matrix(xyz_ens)
            contact_ens_dict[ens_code] = get_contact_map(distance_matrix_ens_dict[ens_code])
        return contact_ens_dict

    def plot_average_dmap_comparison(self, 
                                    ticks_fontsize: int = 14,
                                    cbar_fontsize: int = 14,
                                    title_fontsize: int = 14,
                                    dpi: int = 96,
                                    use_ylabel: bool = True,
                                    save: bool = False):
        """
        Plot the average distance maps for selected ensembles.
        
        Parameters
        ----------
        ticks_fontsize : int, optional
            Font size for tick labels on the plot axes. Default is 14.
        cbar_fontsize : int, optional
            Font size for labels on the color bar. Default is 14.
        title_fontsize : int, optional
            Font size for titles of individual subplots. Default is 14.
        dpi : int, optional
            Dots per inch (resolution) of the output figure. Default is 96.
        use_ylabel : bool, optional
            If True, y-axis labels are displayed on the subplots. Default is True.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        Notes
        -----
        This method plots the average distance maps for selected ensembles, where each distance map
        represents the average pairwise distances between residues in a protein structure.

        """

        ens_dict = self._get_distance_matrix_ens_dict()
        num_proteins = len(ens_dict)
        cols = 2  # Number of columns for subplots
        rows = (num_proteins + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
        axes = axes.reshape((rows, cols))  # Reshape axes to ensure it's 2D
        
        for i, (protein_name, ens_data) in enumerate(ens_dict.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col]# if num_proteins > 1 else axes[0]
            
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

            im.set_clim(0, np.ceil(np.nanmax(avg_dmap.flatten()))) # find the maximum distance and round it to the next integer to manage the auto range
        
        # Remove any empty subplots
        for i in range(num_proteins, rows * cols):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.show()

        self.figures['plot_average_dmap_comparison'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'avg_dmap_' + self.analysis.ens_codes[0]))

        return axes

    def plot_cmap_comparison(self, 
                            title: str,
                            ticks_fontsize: int = 14,
                            cbar_fontsize: int = 14,
                            title_fontsize: int = 14,
                            dpi: int = 96,
                            cmap_min: float = -3.5,
                            use_ylabel: bool = True,
                            save: bool = False):
        """
        Plot the comparison of contact probability maps (Cmaps) for selected ensembles.

        Parameters
        ----------
        title : str
            The title of the plot.
        ticks_fontsize : int, optional
            Font size for tick labels on the plot axes. Default is 14.
        cbar_fontsize : int, optional
            Font size for labels on the color bar. Default is 14.
        title_fontsize : int, optional
            Font size for the main title of the plot. Default is 14.
        dpi : int, optional
            Dots per inch (resolution) of the output figure. Default is 96.
        cmap_min : float, optional
            The minimum value for the color map scale. Default is -3.5.
        use_ylabel : bool, optional
            If True, y-axis labels are displayed on the subplots. Default is True.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        Notes
        -----
        This method plots the comparison of contact probability maps (Cmaps) for selected ensembles.
        Each Cmap represents the log-transformed contact probabilities between residue pairs in a protein
        structure. The comparison is visualized as a grid of subplots, with each subplot displaying the
        Cmap for a specific ensemble.

        """

        cmap_ens_dict = self._get_contact_ens_dict()
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

        self.figures['plot_cmap_comparison'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'cmap_' + self.analysis.ens_codes[0]))
        
        return axes

    def plot_distance_distribution_multiple(self, dpi: int = 96, save: bool = False):
        """
        Plot the distribution of distances for multiple proteins.

        Parameters
        ----------
        dpi : int, optional
            Dots per inch (resolution) of the output figure. Default is 96.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        Notes
        -----
        This method plots the distribution of distances for multiple proteins based on their distance matrices.
        Each subplot in the resulting grid represents the distance distribution for a specific protein.

        The distance distributions are visualized as histograms, where the x-axis represents the distance
        in nanometers and the y-axis represents the density of distances.

        """

        prot_data_dict = self._get_distance_matrix_ens_dict()
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

        self.figures['plot_distance_distribution_multiple'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'distance_distribution_' + self.analysis.ens_codes[0]))
        
        return axes
        

    def end_to_end_distances_plot(self, bins: int = 50, violin_plot: bool = True, means: bool = True, median: bool = True, save: bool = False):
        """
        Plot end-to-end distance distributions.

        Parameters
        ----------
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is True.
        means : bool, optional
            If True, means are shown in the violin plot. Default is True.
        median : bool, optional
            If True, medians are shown in the violin plot. Default is True.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        """
        ensembles = self.analysis.ensembles
        dist_list = []
        positions = []
        
        fig, ax = plt.subplots()

        if violin_plot:
            for ens_code, ensemble in ensembles.items():
                ca_indices = ensemble.trajectory.topology.select(ensemble.atom_selector)
                positions.append(ens_code)
                dist_list.append(mdtraj.compute_distances(ensemble.trajectory, [[ca_indices[0], ca_indices[-1]]]).ravel())
            ax.violinplot(dist_list, showmeans=means, showmedians=median)
            ax.set_xticks(ticks=[y + 1 for y in range(len(positions))])
            ax.set_xticklabels(labels=positions, rotation=45.0, ha="center")
            ax.set_ylabel("End-to-End distance [nm]")
            ax.set_title("End-to-End distances distribution")
        else:
            for ens_code, ensemble in ensembles.items():
                ca_indices = ensemble.trajectory.topology.select(ensemble.atom_selector)
                ax.hist(mdtraj.compute_distances(ensemble.trajectory, [[ca_indices[0], ca_indices[-1]]]).ravel(),
                        label=ens_code, bins=bins, edgecolor='black', density=True)
            ax.set_title("End-to-End distances distribution")
            ax.legend()

        self.figures['end_to_end_distances_plot'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'e2e_distances_' + self.analysis.ens_codes[0]))

        return ax


    def plot_asphericity_dist(self, bins: int = 50, violin_plot: bool = True, means: bool = True, median: bool = True, save: bool = False):
        """
        Plot asphericity distribution in each ensemble.
        Asphericity is calculated based on the gyration tensor.

        Parameters
        ----------
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is True.
        means : bool, optional
            If True, means are shown in the violin plot. Default is True.
        median : bool, optional
            If True, medians are shown in the violin plot. Default is True.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        """
        ensembles = self.analysis.ensembles
        asph_list = []
        positions = []
        
        # Create a new figure object
        fig, ax = plt.subplots()

        if violin_plot:
            for ens_code, ensemble in ensembles.items():
                asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(ensemble.trajectory))
                asph_list.append(asphericity)
                positions.append(ens_code)
            ax.violinplot(asph_list, showmeans=means, showmedians=median)
            ax.set_xticks(ticks=[y + 1 for y in range(len(positions))])
            ax.set_xticklabels(labels=positions, rotation=45.0, ha="center")
            ax.set_ylabel("Asphericity")
            ax.set_title("Asphericity distribution")
        else:
            for ens_code, ensemble in ensembles.items():
                asphericity = calculate_asphericity(mdtraj.compute_gyration_tensor(ensemble.trajectory))
                ax.hist(asphericity, label=ens_code, bins=bins, edgecolor='black', density=True)
            ax.set_title("Asphericity distribution")
            ax.legend()

        self.figures['plot_asphericity_dist'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'asphericity_dist_' + self.analysis.ens_codes[0]))

        return ax

    def plot_prolateness_dist(self, bins: int = 50, violin_plot: bool = True, median: bool = False, mean: bool = False, save: bool =False):
        """
        Plot prolateness distribution in each ensemble.
        Prolateness is calculated based on the gyration tensor.

        Parameters
        ----------
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is True.
        median : bool, optional
            If True, median is showing in the violin plot. Default is False.
        mean : bool, optional
            If True, mean is showing in the violin plot. Default is False.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        """

        ensembles = self.analysis.ensembles
        prolat_list = []
        positions = []

        # Create a new figure object
        fig, ax = plt.subplots()

        if violin_plot:
            for ens_code, ensemble in ensembles.items():
                prolat = calculate_prolateness(mdtraj.compute_gyration_tensor(ensemble.trajectory))
                prolat_list.append(prolat)
                positions.append(ens_code)
            ax.violinplot(prolat_list, showmeans=mean, showmedians=median)
            ax.set_xticks(ticks=[y + 1 for y in range(len(positions))])
            ax.set_xticklabels(labels=positions, rotation=45.0, ha="center")
            ax.set_ylabel("Prolateness")
            ax.set_title("Prolateness distribution")
        else:
            for ens_code, ensemble in ensembles.items():
                prolat = calculate_prolateness(mdtraj.compute_gyration_tensor(ensemble.trajectory))
                ax.hist(prolat, label=ens_code, bins=bins, edgecolor='black', density=True)
            ax.set_title("Prolateness distribution")
            ax.legend()

        self.figures['plot_prolateness_dist'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'prolateness_dist_' + self.analysis.ens_codes[0]))

        return ax


    def plot_alpha_angles_dist(self, bins: int = 50, save: bool = False):
        """
        Plot the distribution of alpha angles.

        Parameters
        ----------
        bins : int
            The number of bins for the histogram. Default is 50.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        """

        ensembles = self.analysis.ensembles

        fig, ax = plt.subplots()

        for ens_code, ensemble in ensembles.items():
            ax.hist(featurize_a_angle(ensemble.trajectory, get_names=False, atom_selector=ensemble.atom_selector).ravel(),
                    bins=bins, histtype="step", density=False, label=ens_code)

        ax.set_title("Distribution of alpha angles")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        self.figures['plot_alpha_angles_dist'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'alpha_dist_' + self.analysis.ens_codes[0]))
        
        return ax


    def plot_contact_prob(self ,norm=True, min_sep=2,max_sep=None ,threshold: float = 0.8, dpi: int = 96, save: bool = False, cmap_color='Blues'):
        from matplotlib.colors import LogNorm
        """
        Plot the contact probability map based on the threshold. 
        The default value for threshold is 0.8[nm], 

        Parameters
        ----------
        title: str 
            You need to specify a title for the plot

        threshold: float
            Determing the threshold fo calculating the contact frequencies. default value is 0.8[nm]

        dpi: int
            For changing the quality and dimension of the output figure

        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        
        norm : bool
            If True, use log scale range

        cmap_color = select a color for the contact map. Default is "Blues"   

        """
        #if self.analysis.exists_coarse_grained():
        #    self._contact_prob_cg( threshold=threshold, dpi=dpi, min_sep=min_sep, max_sep=max_sep)
        #else:
        ensembles = self.analysis.ensembles
        num_proteins = len(ensembles)
        cols = 2
        rows = num_proteins
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
        cmap = cm.get_cmap(cmap_color)
        axes = axes.reshape((rows, cols)) 
        for (ens_code, ensemble) , ax in zip((ensembles.items()), fig.axes):
            if ensemble.coarse_grained:
                matrtix_p_map = contact_probability_map(ensemble.trajectory, scheme='closest', contact=self._pair_ids(min_sep=min_sep, max_sep=max_sep), threshold=threshold)
            else:
                matrtix_p_map = contact_probability_map(ensemble.trajectory, threshold=threshold)

            if norm:
                im = ax.imshow(matrtix_p_map, cmap=cmap, norm=LogNorm() )
            else:
                im = ax.imshow(matrtix_p_map, cmap=cmap )
            ax.set_title(f"Contact Probability Map: {ens_code}", fontsize=14)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Frequency', fontsize=14)
            cbar.ax.tick_params(labelsize=14)

        for i in range(num_proteins, rows * cols):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.show()

        self.figures['plot_contact_prob'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'contact_prob_' + self.analysis.ens_codes[0]))
        
        return axes

    def _pair_ids(self, min_sep=2,max_sep = None ):
        analysis = self.analysis
        pair_ids = []
        for ens in analysis.ensembles:
            ca_ids = analysis.ensembles[ens].trajectory.topology.select('name')
            atoms = list(analysis.ensembles[ens].trajectory.topology.atoms)
            max_sep = get_max_sep(L=len(atoms), max_sep=max_sep)
    # Get all pair of ids.
            for i, id_i in enumerate(ca_ids):
                for j, id_j in enumerate(ca_ids):
                    if j - i >= min_sep:
                        if j - i > max_sep:
                            continue
                        pair_ids.append([id_i, id_j])
        return pair_ids

    '''
    def _contact_prob_cg(self ,min_sep=2,max_sep = None ,threshold = 0.8,dpi = 96, save=False):

        ensembles = self.analysis.ensembles
        num_proteins = len(ensembles)
        cols = 2
        rows = num_proteins
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
        cmap = cm.get_cmap("Blues")
        axes = axes.reshape((rows, cols)) 
        for (ens_code, ensemble) , ax in zip((ensembles.items()), fig.axes):

            matrtix_p_map = contact_probability_map(ensemble.trajectory ,scheme='closest', contact=self._pair_ids(min_sep=min_sep, max_sep=max_sep) ,threshold=threshold)
            im = ax.imshow(matrtix_p_map, cmap=cmap )
            ax.set_title(f"Contact Probability Map: {ens_code}", fontsize=14)


            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Frequency', fontsize=14)
            cbar.ax.tick_params(labelsize=14)

        for i in range(num_proteins, rows * cols):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.show()

        self.figures['plot_contact_prob'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'contact_prob_' + self.analysis.ens_codes[0]))  

    '''
    
    def plot_ramachandran_plot(self, two_d_hist: bool = True, linespaces: Tuple = (-180, 180, 80), save:bool = False):
        
        """
        Ramachandran plot. If two_d_hist= True it returns 2D histogram 
        for each ensembles. If two_d_hist=False it returns a simple scatter plot 
        for ell ensembles in one plot.

        Parameters
        ----------

        two_d_hist: bool, optional
            If True it returns 2D histogram for each ensemble. Default is True.

        linespaces: tuple, optional
            You can customize the bins for 2D histogram. Default is (-180, 180, 80).

        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        """
        
        ensembles = self.analysis.ensembles
        if two_d_hist:
            fig, axes = plt.subplots(1, len(ensembles), figsize=(5*len(ensembles), 5))
            rama_linspace = np.linspace(linespaces[0], linespaces[1], linespaces[2])
            for ens, ax in zip(ensembles, axes):
                phi_flat = np.degrees(mdtraj.compute_phi(ensembles[ens].trajectory)[1]).ravel()
                psi_flat = np.degrees(mdtraj.compute_psi(ensembles[ens].trajectory)[1]).ravel()
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
            fig,axes = plt.subplots(1,1)
            for ens in ensembles:
                phi = np.degrees(mdtraj.compute_phi(ensembles[ens].trajectory)[1])
                psi = np.degrees(mdtraj.compute_psi(ensembles[ens].trajectory)[1])
                plt.scatter(phi, psi, s=1, label= ens)
            axes.set_xlabel('Phi (ϕ) Angle (degrees)')
            axes.set_ylabel('Psi (ψ) Angle (degrees)')
            plt.legend(bbox_to_anchor=(1.04,0), loc = "lower left")
            plt.show()

        self.figures['plot_ramachandran_plot'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'ramachandran_' + self.analysis.ens_codes[0]))  

        return axes

    def plot_ss_measure_disorder(self, pointer: list = None, figsize: Tuple = (15,5), save: bool = False):
        
        """
        Generate site specific flexibility parameter plot. Further information is available in
        this paper by G.Jeschke https://onlinelibrary.wiley.com/doi/epdf/10.1002/pro.4906. 
        In summary this score is sensitive to local flexibility based on the circular variance of the
        Ramachandran angles φ and ψ for each residue in the ensemble.

        This score ranges from 0 for identical dihederal angles for all conformers at the residue i to 1 
        for a uniform distribution of dihederal angles at the residue i. 

        Parameters
        ----------
        pointer: list, optional
            You can add the desired residues in a list and then you have a vertical dashed line to point those residues. Default is None.

        figsize:tuple, optional
            You can change the size of the figure here using a tuple. Default is (15,5).
            
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        """
        if self.analysis.exists_coarse_grained():
            print("This analysis is not possible with coarse-grained models.")
            return
        features_dict = self.analysis.get_features(featurization='phi_psi')
        #for ens in self.analysis.ensembles.values():
        #    features = featurize_phi_psi(traj = ens.trajectory, get_names = False)
        #    features_dict[ens.ens_code] = features

        f = ss_measure_disorder(features_dict)
        fig, axes = plt.subplots(1,1, figsize=figsize)
        keys = list(features_dict.keys())
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

        self.figures['plot_ss_measure_disorder'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'ss_measure_disorder_' + self.analysis.ens_codes[0]))  

        return axes
            
    def plot_ss_order_parameter(self, pointer: list = None , figsize: Tuple = (15,5), save: bool = False): 
        
        """
        Generate site specific order parameter plot. For further information
        you can check this paper by G.Jeschke https://onlinelibrary.wiley.com/doi/epdf/10.1002/pro.4906. 
        In summary this score abstracts from local chain flexibility. The parameter is still site-specific, as orientation 
        correlations in IDRs and IDPs decrease with increasing sequence distance. 

        Parameters
        ----------
        pointer: list, optional
            You can add the desired residues in a list and then you have a vertical dashed line to point those residues. Default is None.

        figsize:tuple, optional
            You can change the size oof the figure here using a tuple. Default is (15,5).
        
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        """
        
        ensembles = self.analysis.ensembles
        dict_ca_xyz = {}
        for ens_code, ensemble in ensembles.items():
            ca_index= ensemble.trajectory.topology.select(ensemble.atom_selector)
            dict_ca_xyz[ens_code] = ensemble.trajectory.xyz[:,ca_index,:]

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

        self.figures['plot_ss_order_parameter'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'ss_order_' + self.analysis.ens_codes[0]))  
        
        return axes

    def plot_local_sasa(self, figsize: Tuple = (15,5), pointer: list = None, save: bool = False): 

        """
        It plots the average SASA for each residue among all conformations in an ensemble.

        Parameters
        ----------
        pointer: list, optional
            You can add the desired residues in a list and then you have a vertical dashed line to point those residues. Default is None.

        figsize:tuple, optional
            You can change the size oof the figure here using a tuple. Default is (15,5).
        
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.

        """

        analysis = self.analysis

        fig, ax = plt.subplots(1,1,figsize=figsize)
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, ens in enumerate(analysis.ensembles):

            res_based_sasa = mdtraj.shrake_rupley(analysis.ensembles[ens].trajectory, mode='residue')
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

        self.figures['plot_local_sasa'] = fig
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'local_sasa_' + self.analysis.ens_codes[0]))  

        return ax

    def plot_dist_ca_com(self, min_sep=2, max_sep=None, get_names=True, inverse=False, figsize=(6, 2.5), save=False):
        """
        Plot the distance maps comparing the center of mass (COM) and alpha-carbon (CA) distances within each ensemble.

        Parameters:
        -----------
        min_sep : int, optional
            Minimum separation distance between atoms to consider. Default is 2.
        max_sep : int, optional
            Maximum separation distance between atoms to consider. Default is None, which means no maximum separation.
        get_names : bool, optional
            Whether to get the residue names for the features. Default is True.
        inverse : bool, optional
            Whether to compute the inverse distances. Default is False.
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (6, 2.5).
        save : bool, optional
            If True, save the plot as an image file. Default is False.

        Notes:
        ------
        This method plots the average distance maps for the center of mass (COM) and alpha-carbon (CA) distances
        within each ensemble. It computes the distance matrices for COM and CA atoms and then calculates their
        mean values to generate the distance maps. The plots include color bars indicating the distance range.

        """

        if self.analysis.exists_coarse_grained():
            print("This analysis is not possible with coarse-grained models.")
            return
        
        axes = []

        analysis = self.analysis
        for ens in analysis.ensembles:
            traj = analysis.ensembles[ens].trajectory
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

            self.figures['plot_dist_ca_com'] = fig
            if save:
                fig.savefig(os.path.join(self.plot_dir, 'dist_ca_com_' + ens))  

            axes.append(ax)

        return axes
    #----------------------------------------------------------------------
    #------------- Functions for generating PDF reports -------------------
    #----------------------------------------------------------------------

    def _generate_tsne_report(self):
        analysis = self.analysis
        pdf_file_path = os.path.join(self.plot_dir, 'tsne.pdf')
        with PdfPages(pdf_file_path) as pdf:
            if analysis.featurization == "phi_psi":
                self.tsne_ramachandran_plot_density(save=False)
                pdf.savefig()
                plt.close()

            self.tsne_scatter_plot(save=False)
            pdf.savefig()
            plt.close()
        
        print(f"Plots saved to {pdf_file_path}")

    def _generate_dimenfix_report(self):
        pdf_file_path = os.path.join(self.plot_dir, 'dimenfix.pdf')
        with PdfPages(pdf_file_path) as pdf:
            self.dimenfix_scatter()
            pdf.savefig()
            plt.close()

        print(f"Plots saved to {pdf_file_path}")

    def _generate_pca_report(self):
        pdf_file_path = os.path.join(self.plot_dir, 'pca.pdf')
        with PdfPages(pdf_file_path) as pdf:
        
            self. pca_cumulative_explained_variance()
            pdf.savefig()
            plt.close()

            self.pca_plot_2d_landscapes()
            pdf.savefig()
            plt.close()

            self.pca_plot_1d_histograms()
            pdf.savefig()
            plt.close()

            self.pca_rg_correlation()
            pdf.savefig()
            plt.close()

        print(f"Plots saved to {pdf_file_path}")

    def generate_custom_report(self):

        """
        Generate pdf report with all plots that were explicitly called during the session.
        """

        analysis = self.analysis
        pdf_file_path = os.path.join(self.plot_dir, 'custom_report.pdf')
        with PdfPages(pdf_file_path) as pdf:
            for fig in self.figures.values():
                pdf.savefig(fig)
                plt.close()
        print(f"Plots saved to {pdf_file_path}")

    def generate_report(self):

        """
        Generate pdf report with all plots relevant to the conducted analysis.
        """

        if self.analysis.reduce_dim_method == "tsne":
            self._generate_tsne_report()
        if self.analysis.reduce_dim_method == "dimenfix":
            self._generate_dimenfix_report()
        if self.analysis.reduce_dim_method == "pca":
            self._generate_pca_report()

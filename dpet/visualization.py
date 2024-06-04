import os
from typing import List, Tuple, Union
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import colors, pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import mdtraj
from dpet.featurization.distances import *
from dpet.ensemble_analysis import EnsembleAnalysis
from dpet.featurization.angles import featurize_a_angle
from dpet.data.coord import *
from dpet.featurization.glob import compute_asphericity, compute_prolateness

PLOT_DIR = "plots"

def plot_histogram(
        ax: plt.Axes,
        data: List[np.ndarray],
        labels: List[str],
        bins: Union[int, List] = 50,
        range: Tuple = None,
        title: str = "Histogram",
        xlabel: str = "x",
        ylabel: str = "Density"
    ):
    """
    Plot an histogram for different features.

    Parameters
    ----------
    ax: plt.Axes
        Matplotlib axis object where the histograms will be for plotted.
    data: List[np.array]
        List of NumPy array storing the data to be plotted.
    labels: List[str]
        List of strings with the labels of the arrays.
    bins:
        Number of bins.
    range: Tuple, optional
        A tuple with a min and max value for the histogram. Default is None,
        which corresponds to using the min a max value across all data.
    title: str, optional
        Title of the axis object.
    xlabel: str, optional
        Label of the horizontal axis.
    ylabel: str, optional
        Label of the vertical axis.

    Returns
    -------
    plt.Axes
        Axis objects for the histogram plot of original labels.
    """
    
    _bins = _get_hist_bins(data=data, bins=bins, range=range)

    for i, data_i in enumerate(data):
        h_i = ax.hist(
            data_i,
            label=labels[i],
            bins=_bins if i == 0 else _bins,
            density=True,
            histtype='step',
            # edgecolor='black',
            # histtype='stepfilled',
            # alpha=0.25
        )
        if i == 0:
            _bins = h_i[1]
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

def _get_hist_bins(data: List[np.ndarray], bins: int, range: Tuple = None):
    if isinstance(bins, int):  # Make a range.
        if range is None:
            _min = min([min(x_i) for x_i in data])
            _max = max([max(x_i) for x_i in data])
        else:
            _min = range[0]
            _max = range[1]
        _bins = np.linspace(_min, _max, bins+1)
    else:  # The bins are already provided as a range.
        _bins = bins
    return _bins

def plot_violins(
        ax: plt.Axes,
        data: List[np.ndarray],
        labels: List[str],
        means: bool = False,
        median: bool = True,
        title: str = "Histogram",
        xlabel: str = "x"
    ):
    """
    Make a violin plot.

    Parameters
    ----------
    ax: plt.Axes
        Matplotlib axis object where the histograms will be for plotted.
    data: List[np.array]
        List of NumPy array storing the data to be plotted.
    labels: List[str]
        List of strings with the labels of the arrays.
    means : bool, optional
            If True, means are shown in the violin plot. Default is True.
    median : bool, optional
        If True, medians are shown in the violin plot. Default is True.
    title: str, optional
        Title of the axis object.
    xlabel: str, optional
        Label of the horizontal axis.

    Returns
    -------
    plt.Axes
        Axis objects for the histogram plot of original labels.
    """
    
    ax.violinplot(data, showmeans=means, showmedians=median)
    ax.set_xticks(ticks=[y + 1 for y in range(len(labels))])
    ax.set_xticklabels(labels=labels, rotation=45.0, ha="center")
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    return ax

def plot_comparison_matrix(
        ax: plt.Axes,
        scores: np.ndarray,
        codes: List[str],
        std: bool = False,
        cmap: str = "viridis_r",
        title: str = "New Comparison",
        cbar_label: str = "score",
        textcolors: Union[str, tuple] = ("black", "white")
    ):
    """
    Plot a matrix with all vs all comparison scores as a heatmap.

    Parameters
    ----------
    ax: plt.Axes
        Axes object where the heatmap should be created.
    scores: np.ndarray
        NumPy array with shape (M, M, B) containing the comparison scores for M
        ensembles and B bootstrap iterations. See the documentation for the
        output of `EnsembleAnalysis.comparison_scores` for more information.
    codes: List[str]
        List of strings with the codes of the ensembles.
    std: bool, optional
        Show the standard deviation of the comparison scores over B bootstrap
        iterations for each of each score in the M x M matrix. Only takes effect
        if B >= 2. Default is `False`.
    cmap: str, optional
        Matplotlib colormap name to use in the heatmap.
    title: str, optional
        Title of the heatmap.
    cbar_label: str, optional
        Label of the colorbar.
    textcolors: Union[str, tuple], optional
        Color of the text for each cell of the heatmap, specified as a string.
        By providing a tuple with two elements, the two colors will be applied
        to cells with color intensity above/below a certain threshold, so that
        ligher text can be plotted in darker cells and darker text can be
        plotted in lighter cells.

    Returns
    -------
    ax: plt.Axes
        The same updated Axes object from the input.
    
    Notes
    -----
    The comparison matrix is annotated with the scores, and the axes are labeled
    with the ensemble labels.

    """
    scores_mean = scores.mean(axis=2)
    if scores.shape[2] > 1 and std:
        # More than 1 bootstrap iteration, can calculate std.
        scores_err = scores.std(axis=2)
    else:
        # One or less bootstrap iterations.
        scores_err = None

    ax.set_title(title)
    im = ax.imshow(scores_mean, cmap=cmap)
    plt.colorbar(im, label=cbar_label)
    ax.set_xticks(np.arange(len(codes)))
    ax.set_yticks(np.arange(len(codes)))
    ax.set_xticklabels(codes, rotation=45)
    ax.set_yticklabels(codes)

    threshold = 0.75
    for i in range(len(codes)):
        for j in range(len(codes)):
            if isinstance(textcolors, str):
                color_ij = textcolors
            else:
                color_ij = textcolors[int(im.norm(scores_mean[i, j]) > threshold)]
            kw = {"color": color_ij}
            if scores_err is not None:
                label_ij = f"{scores_mean[i, j]:.3f} ± {scores_err[i, j]:.3f}"
            else:
                label_ij = f"{scores_mean[i, j]:.3f}"
            text = im.axes.text(j, i, label_ij, ha="center", va="center", **kw)

    return ax


class Visualization:
    """
    Visualization class for ensemble analysis.

    Parameters:
        analysis (EnsembleAnalysis): An instance of EnsembleAnalysis providing data for visualization.
    """

    def __init__(self, analysis: EnsembleAnalysis):
        self.analysis = analysis
        self.plot_dir = os.path.join(self.analysis.output_dir, PLOT_DIR)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _tsne_scatter(
            self,
            color_by: str = "rg",
            kde_by_ensemble: bool = False,
            save: bool = False,
            ax: Union[None, plt.Axes, np.ndarray, List[plt.Axes]] = None
    ) -> List[plt.Axes]:
        """
        Plot the results of t-SNE analysis. 

        Three scatter plots will be generated based on original, clustering, and feature-colored points. 
        One KDE density plot will also be generated to show the most populated areas in the reduced dimension.   

        Parameters
        ----------
        color_by: str, optional
            The feature extraction method used for coloring points in the scatter plot. Options are "rg", "prolateness", "asphericity", "sasa", and "end_to_end". Default is "rg".
        
        kde_by_ensemble: bool, optional
            If True, the KDE plot will be generated for each ensemble separately. If False, a single KDE plot will be generated for the concatenated ensembles. Default is False.
        
        save: bool, optional
            If True, the plot will be saved in the data directory. Default is False.
        
        ax: Union[None, plt.Axes, np.ndarray, List[plt.Axes]], optional
            The axes on which to plot. If None, new axes will be created. Default is None.

        Returns
        -------
        List[plt.Axes]
            List containing Axes objects for the scatter plot of original labels, clustering labels, feature-colored labels, and the KDE density plot, respectively.

        Notes
        ------
        This analysis is only valid for t-SNE dimensionality reduction.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method != "tsne":
            raise ValueError("Analysis is only valid for t-SNE dimensionality reduction.")
        
        if color_by not in ("rg", "prolateness", "asphericity", "sasa", "end_to_end"):
            raise ValueError(f"Method {color_by} not supported.")

        bestclust = analysis.reducer.best_kmeans.labels_
        
        if ax is None:
            fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        else:
            if not isinstance(ax, (list, np.ndarray)):
                ax = [ax]
            ax = np.array(ax).flatten()
            fig = ax[0].figure
            
        # Create a consistent colormap for the original labels
        unique_labels = np.unique(analysis.all_labels)
        cmap = plt.get_cmap('plasma')
        colors = cmap(np.linspace(0, 1, len(unique_labels)))
        label_colors = {label: color for label, color in zip(unique_labels, colors)}
        point_colors = [label_colors[label] for label in analysis.all_labels]

        # Scatter plot with original labels
        scatter_labeled = ax[0].scatter(analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1], c=point_colors, s=10, alpha=0.5)
        ax[0].set_title('Scatter plot (original labels)')

        # Scatter plot with clustering labels
        cmap = plt.get_cmap('jet', analysis.reducer.bestK)
        scatter_cluster = ax[1].scatter(analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1], s=10, c=bestclust.astype(float), cmap=cmap, alpha=0.5)
        ax[1].set_title('Scatter plot (clustering labels)')

        feature_values = []
        for values in analysis.get_features(color_by).values():
            feature_values.extend(values)
        colors = np.array(feature_values)

        feature_labeled = ax[2].scatter(analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1], c=colors, s=10, alpha=0.5)
        cbar = plt.colorbar(feature_labeled, ax=ax[2])
        ax[2].set_title(f'Scatter plot ({color_by} labels)')

        if kde_by_ensemble:
            # KDE plot for each ensemble
            for label in unique_labels:
                ensemble_data = analysis.reducer.best_tsne[np.array(analysis.all_labels) == label]
                kde = gaussian_kde([ensemble_data[:, 0], ensemble_data[:, 1]])
                xi, yi = np.mgrid[min(ensemble_data[:, 0]):max(ensemble_data[:, 0]):100j,
                                min(ensemble_data[:, 1]):max(ensemble_data[:, 1]):100j]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                ax[3].contour(xi, yi, zi.reshape(xi.shape), levels=5, alpha=0.5, label=f'Ensemble {label}', colors=[label_colors[label]])
            ax[3].set_title('Density Plot (Ensemble-wise)')
            ax[3].legend(title='Ensemble', loc='upper right')
        else:
            # Single KDE plot for concatenated ensembles
            kde = gaussian_kde([analysis.reducer.best_tsne[:, 0], analysis.reducer.best_tsne[:, 1]])
            xi, yi = np.mgrid[min(analysis.reducer.best_tsne[:, 0]):max(analysis.reducer.best_tsne[:, 0]):100j,
                            min(analysis.reducer.best_tsne[:, 1]):max(analysis.reducer.best_tsne[:, 1]):100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            ax[3].contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Blues')
            ax[3].set_title('Density Plot')

        # Manage legend for the original labels
        legend_labels = list(label_colors.keys())
        legend_handles = [plt.Line2D([0], [0], marker='o', color=label_colors[label], markersize=10) for label in legend_labels]
        fig.legend(legend_handles, legend_labels, title='Original Labels', loc='upper right')

        fig.tight_layout()

        if save:
            fig.savefig(self.plot_dir + f'/tsnep{int(analysis.reducer.bestP)}_kmeans{int(analysis.reducer.bestK)}_scatter.png', dpi=800)

        return ax
    
    def dimensionality_reduction_scatter(self,
                                         color_by: str = "rg", 
                                         save: bool = False, 
                                         ax: Union[None, List[plt.Axes]] = None,
                                         kde_by_ensemble: bool = False) -> List[plt.Axes]:
        """
        Plot the results of dimensionality reduction using the method specified in the analysis.

        Parameters
        ----------
        color_by : str, optional
            The feature extraction method used for coloring points in the scatter plot. 
            Options are "rg", "prolateness", "asphericity", "sasa", and "end_to_end". Default is "rg".

        save : bool, optional
            If True, the plot will be saved in the data directory. Default is False.

        ax : Union[None, List[plt.Axes]], optional
            A list of Axes objects to plot on. Default is None, which creates new axes.

        kde_by_ensemble : bool, optional
            If True, the KDE plot will be generated for each ensemble separately. 
            If False, a single KDE plot will be generated for the concatenated ensembles. Default is False.

        Returns
        -------
        List[plt.Axes]
            List containing Axes objects for the scatter plot of original labels, clustering labels, and feature-colored labels, respectively.

        Raises
        ------
        NotImplementedError
            If the dimensionality reduction method specified in the analysis is not supported.

        """

        method = self.analysis.reduce_dim_method
        if method in ("dimenfix", "umap"):
            self._dimenfix_umap_scatter(color_by=color_by, save=save, ax=ax, kde_by_ensemble=kde_by_ensemble)
        elif method == "tsne":
            self._tsne_scatter(color_by=color_by, kde_by_ensemble=kde_by_ensemble, save=save, ax=ax)
        else:
            raise NotImplementedError(f"Scatter plot for method '{method}' is not implemented. Please select between 'tsne', 'dimenfix', and 'umap'.")

    def _dimenfix_umap_scatter(self, 
                         color_by: str = "rg", 
                         save: bool = False, 
                         ax: Union[None, List[plt.Axes]] = None,
                         kde_by_ensemble: bool = False,
                         ) -> List[plt.Axes]:
        """
        Plot the complete results for dimenfix and umap methods. 

        Parameters
        -----------
        color_by: str, optional
            The feature extraction method used for coloring points in the scatter plot. Options are "rg", "prolateness", "asphericity", "sasa", and "end_to_end". Default is "rg".

        save: bool, optional
            If True, the plot will be saved in the data directory. Default is False.

        ax : Union[None, List[plt.Axes]], optional
            A list of Axes objects to plot on. Default is None, which creates new axes.
        
        kde_by_ensemble: bool, optional
            If True, the KDE plot will be generated for each ensemble separately. If False, a single KDE plot will be generated for the concatenated ensembles. Default is False.

        Returns
        --------
        List[plt.Axes]
            List containing Axes objects for the scatter plot of original labels, clustering labels, and feature-colored labels, respectively.

        """

        analysis = self.analysis

        if analysis.reduce_dim_method not in ("dimenfix", "umap"):
            raise ValueError("Analysis is only valid for dimenfix dimensionality reduction.")
        
        if color_by not in ("rg", "prolateness", "asphericity", "sasa", "end_to_end"):
            raise ValueError(f"Method {color_by} not supported.")

        if ax is None:
            fig, ax = plt.subplots(1, 4, figsize=(18, 4))
            axes = ax.flatten()  # Ensure axes is a 1D array
        else:
            ax_array = np.array(ax).flatten()
            axes = ax_array  # If ax is provided, flatten it to 1D
            fig = axes[0].figure

        # Create a consistent colormap for the original labels
        unique_labels = np.unique(analysis.all_labels)
        cmap = plt.get_cmap('plasma')
        colors = cmap(np.linspace(0, 1, len(unique_labels)))
        label_colors = {label: color for label, color in zip(unique_labels, colors)}
        point_colors = [label_colors[label] for label in analysis.all_labels]

        # Scatter plot with original labels
        scatter_labeled = axes[0].scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], c=point_colors, s=10, alpha=0.5)
        axes[0].set_title('Scatter plot (original labels)')

        # Scatter plot with different labels
        feature_values = []
        for values in analysis.get_features(color_by).values():
            feature_values.extend(values)
        colors = np.array(feature_values)
        
        rg_labeled = axes[2].scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], c=colors, s=10, alpha=0.5)
        cbar = plt.colorbar(rg_labeled, ax=axes[2])
        axes[2].set_title(f'Scatter plot ({color_by} labels)')

        # Scatter plot with clustering labels
        best_k = max(analysis.reducer.sil_scores, key=lambda x: x[1])[0]
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(analysis.transformed_data)
        scatter_cluster = axes[1].scatter(analysis.transformed_data[:, 0], analysis.transformed_data[:, 1], s=10, c=labels, cmap='viridis')
        axes[1].set_title('Scatter plot (clustering labels)')

        # Manage legend for original labels
        legend_labels = list(label_colors.keys())
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], markersize=10) for label in legend_labels]
        fig.legend(legend_handles, legend_labels, title='Original Labels', loc='upper right')

        if kde_by_ensemble:
            # KDE plot for each ensemble
            for label in unique_labels:
                ensemble_data = analysis.transformed_data[np.array(analysis.all_labels) == label]
                kde = gaussian_kde([ensemble_data[:, 0], ensemble_data[:, 1]])
                xi, yi = np.mgrid[min(ensemble_data[:, 0]):max(ensemble_data[:, 0]):100j,
                                min(ensemble_data[:, 1]):max(ensemble_data[:, 1]):100j]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                ax[3].contour(xi, yi, zi.reshape(xi.shape), levels=5, alpha=0.5, label=f'Ensemble {label}', colors=[label_colors[label]])
            ax[3].set_title('Density Plot (Ensemble-wise)')
            ax[3].legend(title='Ensemble', loc='upper right')
        else:
            # Single KDE plot for concatenated ensembles
            kde = gaussian_kde([analysis.transformed_data[:, 0], analysis.transformed_data[:, 1]])
            xi, yi = np.mgrid[min(analysis.transformed_data[:, 0]):max(analysis.transformed_data[:, 0]):100j,
                            min(analysis.transformed_data[:, 1]):max(analysis.transformed_data[:, 1]):100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            ax[3].contour(xi, yi, zi.reshape(xi.shape), levels=5, cmap='Blues')
            ax[3].set_title('Density Plot')

        if save:
            fig.savefig(self.plot_dir + f'/{analysis.reduce_dim_method}_scatter.png', dpi=800)

        return axes

    def pca_cumulative_explained_variance(self, save: bool = False, ax: Union[None, plt.Axes] = None) -> plt.Axes:
        """
        Plot the cumulative variance. Only applicable when the
        dimensionality reduction method is "pca".

        Parameters
        ----------
        save: bool, optional
            If True, the plot will be saved in the data directory. Default is False.

        ax: Union[None, plt.Axes], optional
            An Axes object to plot on. Default is None, which creates a new axes.

        Returns
        -------
        plt.Axes
            The Axes object for the cumulative explained variance plot.
        """ 
        
        analysis = self.analysis

        if analysis.reduce_dim_method != "pca":
            raise ValueError("Analysis is only valid for pca dimensionality reduction.")
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(np.cumsum(analysis.reduce_dim_model.explained_variance_ratio_) * 100)
        ax.set_xlabel("PCA dimension")
        ax.set_ylabel("Cumulative explained variance %")
        ax.set_title("Cumulative Explained Variance by PCA Dimension")
        ax.grid(True)
        first_three_variance = analysis.reduce_dim_model.explained_variance_ratio_[0:3].sum() * 100
        ax.text(0.5, 0.9, f"First three: {first_three_variance:.2f}%", transform=ax.transAxes, ha='center')

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'PCA_variance' + analysis.featurization + analysis.ens_codes[0]))

        return ax

    def _set_labels(self, ax, reduce_dim_method, dim_x, dim_y):
        ax.set_xlabel(f"{reduce_dim_method} dim {dim_x+1}")
        ax.set_ylabel(f"{reduce_dim_method} dim {dim_y+1}")

    def pca_2d_landscapes(self, save: bool = False, ax: Union[None, List[plt.Axes]] = None) -> List[plt.Axes]:
        """
        Plot 2D landscapes when the dimensionality reduction method is "pca" or "kpca".

        Parameters
        ----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.

        ax: Union[None, List[plt.Axes]], optional
            A list of Axes objects to plot on. Default is None, which creates new axes.

        Returns
        -------
        List[plt.Axes]
            A list of plt.Axes objects representing the subplots created.
        """
        
        analysis = self.analysis

        if analysis.reduce_dim_method not in ("pca", "kpca"):
            raise ValueError("Analysis is only valid for pca or kpca dimensionality reduction.")

        # 2D scatter plot settings
        dim_x = 0
        dim_y = 1
        marker = "."
        legend_kwargs = {"loc": 'upper right', "bbox_to_anchor": (1.1, 1.1), "fontsize": 8}

        num_ensembles = len(analysis.ens_codes)
        
        if ax is None:
            fig, axes = plt.subplots(num_ensembles + 1, figsize=(4, 4 * (num_ensembles + 1)), dpi=120)
        else:
            if not isinstance(ax, (list, np.ndarray)):
                ax = [ax]
            axes = np.array(ax).flatten()
            fig = axes[0].figure

        # Plot all ensembles at the same time
        axes[0].set_title("all")
        for ensemble in analysis.ensembles:
            axes[0].scatter(ensemble.reduce_dim_data[:, dim_x],
                            ensemble.reduce_dim_data[:, dim_y],
                            label=ensemble.code, marker=marker)
        axes[0].legend(**legend_kwargs)
        self._set_labels(axes[0], "pca", dim_x, dim_y)

        # Concatenate all reduced dimensionality data from the dictionary
        all_data = analysis.transformed_data

        # Plot each ensemble individually
        for i, ensemble in enumerate(analysis.ensembles):
            axes[i + 1].set_title(ensemble.code)
            # Plot all data in gray
            axes[i + 1].scatter(all_data[:, dim_x],
                                all_data[:, dim_y],
                                label="all", color="gray", alpha=0.25,
                                marker=marker)
            # Plot ensemble data in color
            axes[i + 1].scatter(ensemble.reduce_dim_data[:, dim_x],
                                ensemble.reduce_dim_data[:, dim_y],
                                label=ensemble.code, c=f"C{i}",
                                marker=marker)
            axes[i + 1].legend(**legend_kwargs)
            self._set_labels(axes[i + 1], "pca", dim_x, dim_y)

        fig.tight_layout()

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'PCA_2d_landscapes_' + analysis.featurization + analysis.ens_codes[0]))

        return axes

    def pca_1d_histograms(self, save: bool = False, ax: Union[None, List[plt.Axes]] = None) -> List[plt.Axes]:
        """
        Plot 1D histogram when the dimensionality reduction method is "pca" or "kpca".

        Parameters
        ----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.

        ax: Union[None, List[plt.Axes]], optional
            A list of Axes objects to plot on. Default is None, which creates new axes.

        Returns
        -------
        List[plt.Axes]
            A list of plt.Axes objects representing the subplots created.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method not in ("pca", "kpca"):
            raise ValueError("Analysis is only valid for pca and kpca dimensionality reduction.")

        n_bins = 30
        k = 0
        bins = np.linspace(analysis.transformed_data[:, k].min(),
                        analysis.transformed_data[:, k].max(),
                        n_bins)

        if ax is None:
            fig, axes = plt.subplots(len(analysis.ens_codes), 1, figsize=(4, 2 * len(analysis.ens_codes)), dpi=120)
        else:
            if not isinstance(ax, (list, np.ndarray)):
                ax = [ax]
            axes = np.array(ax).flatten()
            fig = axes[0].figure

        # Plot histograms for each ensemble
        for i, ensemble in enumerate(analysis.ensembles):
            axes[i].hist(ensemble.reduce_dim_data[:, k],
                        label=ensemble.code,
                        bins=bins,
                        density=True,
                        color=f"C{i}",
                        histtype="step")
            axes[i].hist(analysis.transformed_data[:, k],
                        label="all",
                        bins=bins,
                        density=True,
                        color="gray",
                        alpha=0.25,
                        histtype="step")
            axes[i].legend(loc='upper right',
                        bbox_to_anchor=(1.1, 1.1),
                        fontsize=8)
            axes[i].set_xlabel(f"Dim {k+1}")
            axes[i].set_ylabel("Density")

        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'PCA_hist' + analysis.featurization + analysis.ens_codes[0]))

        return axes

    def pca_residue_correlation(self, sel_dims: List[int], save: bool = False, ax: Union[None, List[plt.Axes]] = None) -> List[plt.Axes]:
        """
        Plot the correlation between residues based on PCA weights.

        Parameters
        ----------
        sel_dims : List[int]
            A list of indices specifying the PCA dimensions to include in the plot.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : Union[None, List[plt.Axes]], optional
            A list of Axes objects to plot on. Default is None, which creates new axes.

        Returns
        -------
        List[plt.Axes]
            A list of plt.Axes objects representing the subplots created.

        Notes
        -----
        This method generates a correlation plot showing the weights of pairwise residue distances
        for selected PCA dimensions. The plot visualizes the correlation between residues based on
        the PCA weights.

        The analysis is only valid on PCA and kernel PCA dimensionality reduction with 'ca_dist' feature extraction.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method != "pca" or analysis.featurization != "ca_dist":
            raise ValueError("Analysis is only valid for pca dimensionality reduction with ca_dist feature extraction.")
        
        cmap = plt.get_cmap("RdBu")  # RdBu, PiYG
        norm = colors.Normalize(-0.07, 0.07)  # NOTE: this range should be adapted when analyzing other systems via PCA!
        dpi = 120

        fig_r = 0.8
        if ax is None:
            fig, axes = plt.subplots(1, 3, dpi=dpi, figsize=(15*fig_r, 4*fig_r))
        else:
            if not isinstance(ax, (list, np.ndarray)):
                ax = [ax]
            axes = np.array(ax).flatten()
            fig = axes[0].figure

        # Get the number of residues from one of the trajectories
        num_residues = next(iter(analysis.trajectories.values())).topology.n_residues

        for k, sel_dim in enumerate(sel_dims):
            feature_ids_sorted_by_weight = np.flip(np.argsort(abs(analysis.reduce_dim_model.components_[sel_dim,:])))
            matrix = np.zeros((num_residues, num_residues))
            for i in feature_ids_sorted_by_weight:
                r1, r2 = analysis.feature_names[i].split("-")
                # Note: this should be patched for proteins with resSeq values not starting from 1!
                matrix[int(r1[3:])-1, int(r2[3:])-1] = analysis.reduce_dim_model.components_[sel_dim,i]
                matrix[int(r2[3:])-1, int(r1[3:])-1] = analysis.reduce_dim_model.components_[sel_dim,i]
            im = axes[k].imshow(matrix, cmap=cmap, norm=norm)  # RdBu, PiYG
            axes[k].set_xlabel("Residue j")
            axes[k].set_ylabel("Residue i")
            axes[k].set_title(r"Weight of $d_{ij}$" + f" for PCA dim {sel_dim+1}")
            cbar = fig.colorbar(
                im, ax=axes[k],
                label="PCA weight"
            )
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'PCA_correlation' + analysis.featurization + analysis.ens_codes[0]))

        return axes

    def pca_rg_correlation(self, save: bool = False, ax: Union[None, List[plt.Axes]] = None) -> List[plt.Axes]:
        """
        Examine and plot the correlation between PC dimension 1 and the amount of Rg.
        Typically high correlation can be detected here.

        Parameters
        ----------
        save : bool, optional
            If True, the plot will be saved in the data directory. Default is False.

        ax: Union[None, List[plt.Axes]], optional
            A list of Axes objects to plot on. Default is None, which creates new axes.

        Returns
        -------
        List[plt.Axes]
            A list of plt.Axes objects representing the subplots created.
        """

        analysis = self.analysis

        if analysis.reduce_dim_method not in ("pca", "kpca"):
            raise ValueError("Analysis is only valid for pca and kpca dimensionality reduction.")

        pca_dim = 0

        if ax is None:
            fig, axes = plt.subplots(len(analysis.ens_codes), 1, figsize=(3, 3 * len(analysis.ens_codes)), dpi=120)
        else:
            if not isinstance(ax, (list, np.ndarray)):
                ax = [ax]
            axes = np.array(ax).flatten()
            fig = axes[0].figure

        # Plot the correlation for each ensemble
        for i, ensemble in enumerate(analysis.ensembles):
            rg_i = mdtraj.compute_rg(ensemble.trajectory)
            axes[i].scatter(ensemble.reduce_dim_data[:, pca_dim],
                            rg_i, label=ensemble.code,
                            color=f"C{i}")
            axes[i].legend(fontsize=8)
            axes[i].set_xlabel(f"Dim {pca_dim + 1}")
            axes[i].set_ylabel("Rg")

        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'PCA_RG' + analysis.ens_codes[0]))

        return axes

    def ensemble_sasa(self, 
                      bins: int = 50, 
                      hist_range: Tuple = None, 
                      violin_plot: bool = True, 
                      means: bool = True, 
                      medians: bool = True, 
                      save: bool = False, 
                      ax: Union[None, plt.Axes] = None) -> plt.Axes:
        """
        Plot the distribution of SASA for each conformation within the ensembles.

        Parameters
        ----------
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        hist_range: Tuple, optional
            A tuple with a min and max value for the histogram. Default is None,
            which corresponds to using the min a max value across all data.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is True.
        means : bool, optional
            If True, it will show the mean. Default is True.
        medians : bool, optional
            If True, it will show the median. Default is True.
        save : bool, optional
            If True, the plot will be saved in the data directory. Default is False.
        ax : Union[None, plt.Axes], optional
            The matplotlib Axes object on which to plot. If None, a new Axes object will be created. Default is None.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        if self.analysis.exists_coarse_grained():
            raise ValueError("This analysis is not possible with coarse-grained models.")

        ensembles = self.analysis.ensembles

        # Calculate features.
        hist_data = []
        labels = []

        for ensemble in ensembles:
            sasa_i = mdtraj.shrake_rupley(ensemble.trajectory)
            total_sasa_i = sasa_i.sum(axis=1)
            hist_data.append(total_sasa_i)
            labels.append(ensemble.code)

        # Plot.
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        axis_label = r"SASA (nm$^2$)"
        title = "SASA distribution over the ensembles"

        if violin_plot:
            plot_violins(
                ax=ax,
                data=hist_data,
                labels=labels,
                means=means,
                median=medians,
                title=title,
                xlabel=axis_label
            )
        else:
            plot_histogram(
                ax=ax,
                data=hist_data,
                labels=labels,
                bins=bins,
                range=hist_range,
                title=title,
                xlabel=axis_label
            )

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'Global_SASA_dist' + self.analysis.ens_codes[0]))

        return ax

    def rg_vs_asphericity(self, save: bool = False, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the Rg versus Asphericity and get the pearson correlation coefficient to evaluate 
        the correlation between Rg and Asphericity.

        Parameters
        ----------
        save: bool, optional
            If True, the plot will be saved in the data directory. Default is False.
        ax: plt.Axes, optional
            The axes on which to plot. Default is None, which creates a new figure and axes.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        analysis = self.analysis
        
        if ax is None:
            fig, ax = plt.subplots()  # Create a new figure if ax is not provided
        else:
            fig = ax.figure  # Use the figure associated with the provided ax
        
        for ensemble in analysis.ensembles:
            x = mdtraj.compute_rg(ensemble.trajectory)
            y = compute_asphericity(ensemble.trajectory)
            p = np.corrcoef(x, y)
            ax.scatter(x, y, s=4, label=ensemble.code)
            print(f"Pearson coeff for {ensemble.code} = {round(p[0][1], 3)}")
        
        ax.set_ylabel("Asphericity")
        ax.set_xlabel("Radius of Gyration (Rg) [nm]")
        ax.legend()
        
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'Rg_vs_Asphericity_' + analysis.ens_codes[0]))
        
        return ax
  
    def rg_vs_prolateness(self, save: bool = False, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the Rg versus Prolateness and get the Pearson correlation coefficient to evaluate 
        the correlation between Rg and Prolateness. 

        Parameters
        ----------
        save: bool, optional
            If True, the plot will be saved in the data directory. Default is False.
        ax: plt.Axes, optional
            The axes on which to plot. Default is None, which creates a new figure and axes.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        analysis = self.analysis
        
        if ax is None:
            fig, ax = plt.subplots()  # Create a new figure if ax is not provided
        else:
            fig = ax.figure  # Use the figure associated with the provided ax

        for ensemble in analysis.ensembles:
            x = mdtraj.compute_rg(ensemble.trajectory)
            y = compute_prolateness(ensemble.trajectory)
            p = np.corrcoef(x, y)
            ax.scatter(x, y, s=4, label=ensemble.code)
            print(f"Pearson coeff for {ensemble.code} = {round(p[0][1], 3)}")

        ax.set_ylabel("Prolateness")
        ax.set_xlabel("Radius of Gyration (Rg) [nm]")
        ax.legend()

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'Rg_vs_Prolateness_' + analysis.ens_codes[0]))
        
        return ax

    def _get_protein_dssp_data_dict(self):
        ensembles = self.analysis.ensembles
        dssp_data_dict = {}
        for ensemble in ensembles:
            dssp_data_dict[ensemble.code] = mdtraj.compute_dssp(ensemble.trajectory)
        return dssp_data_dict
    
    def relative_helix_content(self, save: bool = False, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the relative helix content in each ensemble for each residue. 

        Parameters
        ----------
        save : bool, optional
            If True, the plot will be saved in the data directory. Default is False.
        ax : plt.Axes, optional
            The axes on which to plot. Default is None, which creates a new figure and axes.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        if self.analysis.exists_coarse_grained():
            raise ValueError("This analysis is not possible with coarse-grained models.")
        
        protein_dssp_data_dict = self._get_protein_dssp_data_dict()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        bottom = np.zeros(max(data.shape[1] for data in protein_dssp_data_dict.values()))
        max_length = len(bottom)

        for protein_name, dssp_data in protein_dssp_data_dict.items():
            # Count the occurrences of 'H' in each column
            h_counts = np.count_nonzero(dssp_data == 'H', axis=0)
            
            # Calculate the total number of residues for each position
            total_residues = dssp_data.shape[0]
            
            # Calculate the relative content of 'H' for each residue
            relative_h_content = h_counts / total_residues

            # Interpolate or pad the relative content to ensure all ensembles have the same length
            if len(relative_h_content) < max_length:
                relative_h_content = np.pad(relative_h_content, (0, max_length - len(relative_h_content)), mode='constant')
            
            # Plot the relative content for each protein
            x = np.arange(len(relative_h_content))
            mask = x < len(dssp_data[0])  # Create a mask to filter out padded values
            ax.plot(x[mask], relative_h_content[mask], marker='o', linestyle='dashed', label=protein_name, alpha=0.5)

            bottom += relative_h_content
        
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Relative Content of H (Helix)')
        ax.set_title('Relative Content of H in Each Residue in the ensembles')
        ax.legend()

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'relative_helix_' + self.analysis.ens_codes[0]))
        
        return ax

    def _get_rg_data_dict(self):
        ensembles = self.analysis.ensembles
        rg_dict = {}
        for ensemble in ensembles:
            rg_dict[ensemble.code] = mdtraj.compute_rg(ensemble.trajectory)
        return rg_dict

    def radius_of_gyration(
            self,
            bins: int = 50,
            hist_range: Tuple = None,
            multiple_hist_ax: bool = False,
            violin_plot: bool = False,
            median: bool = False,
            means: bool = False,
            dpi: int = 96,
            save: bool = False,
            ax: Union[None, plt.Axes, np.ndarray, List[plt.Axes]] = None
        ) -> Union[plt.Axes, List[plt.Axes]]:
        """
        Plot the distribution of the radius of gyration (Rg) within each ensemble.

        Parameters
        ----------
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        hist_range : Tuple, optional
            A tuple with a min and max value for the histogram. Default is None,
            which corresponds to using the min and max value across all data.
        multiple_hist_ax: bool, optional
            If True, it will plot each histogram in a different axis.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is False.
        median : bool, optional
            If True, median is shown in the plot. Default is False.
        means : bool, optional
            If True, mean is shown in the plot. Default is False.
        dpi : int, optional
            The DPI (dots per inch) of the output figure. Default is 96.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : Union[None, plt.Axes, np.ndarray, List[plt.Axes]], optional
            The axes on which to plot. If None, new axes will be created. Default is None.

        Returns
        -------
        Union[plt.Axes, List[plt.Axes]]
            Returns a single Axes object or a list of Axes objects containing the plot(s).

        Notes
        -----
        This method plots the distribution of the radius of gyration (Rg) within each ensemble in the analysis.

        The Rg values are binned according to the specified number of bins (`bins`) and range (`hist_range`) and 
        displayed as histograms. Additionally, dashed lines representing the mean and median Rg values are overlaid
        on each histogram.
        """

        # Calculate features.
        rg_data_dict = self._get_rg_data_dict()
        hist_data = list(rg_data_dict.values())
        labels = list(rg_data_dict.keys())
        n_systems = len(rg_data_dict)

        # Plot.
        if not violin_plot and multiple_hist_ax:
            # One axis for each histogram.
            if ax is None:
                fig, ax = plt.subplots(
                    1, n_systems,
                    figsize=(3 * n_systems, 3),
                    dpi=dpi
                )
            else:
                if not isinstance(ax, (list, np.ndarray)):
                    ax = [ax]
                ax = np.array(ax).flatten()
                fig = ax[0].figure
        else:
            # Only one axis for all histograms.
            if ax is None:
                fig, ax = plt.subplots(dpi=dpi)
            else:
                fig = ax.figure

        axis_label = "Radius of Gyration (Rg)"
        title = "Radius of Gyration"

        if violin_plot:
            plot_violins(
                ax=ax,
                data=hist_data,
                labels=labels,
                means=means,
                median=median,
                title=title,
                xlabel=axis_label
            )
        else:
            if not multiple_hist_ax:
                plot_histogram(
                    ax=ax,
                    data=hist_data,
                    labels=labels,
                    bins=bins,
                    range=hist_range,
                    title=title,
                    xlabel=axis_label
                )
            else:
                _bins = _get_hist_bins(
                    data=hist_data, bins=bins, range=hist_range
                )
                h_args = {"histtype": "step", "density": True}

                if isinstance(ax, np.ndarray):
                    ax = ax.flatten()

                for i, (name_i, rg_i) in enumerate(rg_data_dict.items()):
                    ax[i].hist(rg_i, bins=_bins, label=name_i, **h_args)
                    ax[i].set_title(name_i)
                    if i == 0:
                        ax[i].set_ylabel("Density")
                    ax[i].set_xlabel(axis_label)
                    legend_handles = []
                    if means:
                        mean_rg = np.mean(rg_i)
                        mean_line = ax[i].axvline(mean_rg, color='k', linestyle='dashed', linewidth=1)
                        mean_legend = Line2D([0], [0], color='k', linestyle='dashed', linewidth=1, label='Mean')
                        legend_handles.append(mean_legend)
                    if median:
                        median_rg = np.median(rg_i)
                        median_line = ax[i].axvline(median_rg, color='r', linestyle='dashed', linewidth=1)
                        median_legend = Line2D([0], [0], color='r', linestyle='dashed', linewidth=1, label='Median')
                        legend_handles.append(median_legend)
                    if legend_handles:
                        ax[i].legend(handles=legend_handles, loc='upper right')

                    fig.tight_layout()

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'rg_comparison_' + self.analysis.ens_codes[0]))

        return ax

    def _get_distance_matrix_ens_dict(self):
        ensembles = self.analysis.ensembles
        distance_matrix_ens_dict = {}
        for ensemble in ensembles:
            selector = ensemble.atom_selector
            trajectory = ensemble.trajectory
            xyz_ens = trajectory.xyz[:,trajectory.topology.select(selector)]
            distance_matrix_ens_dict[ensemble.code] = get_distance_matrix(xyz_ens)
        return distance_matrix_ens_dict

    def _get_contact_ens_dict(self):
        ensembles = self.analysis.ensembles
        distance_matrix_ens_dict = {}
        contact_ens_dict = {}
        for ensemble in ensembles:
            xyz_ens = ensemble.trajectory.xyz[:,ensemble.trajectory.topology.select(ensemble.atom_selector)]
            distance_matrix_ens_dict[ensemble.code] = get_distance_matrix(xyz_ens)
            contact_ens_dict[ensemble.code] = get_contact_map(distance_matrix_ens_dict[ensemble.code])
        return contact_ens_dict

    def average_distance_maps(self, 
                            ticks_fontsize: int = 14,
                            cbar_fontsize: int = 14,
                            title_fontsize: int = 14,
                            dpi: int = 96,
                            use_ylabel: bool = True,
                            save: bool = False,
                            ax: Union[None, List[List[plt.Axes]], List[plt.Axes]] = None) -> List[plt.Axes]:
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
        ax : Union[None, List[List[plt.Axes]], List[plt.Axes]], optional
            A list or 2D list of Axes objects to plot on. Default is None, which creates new axes.

        Returns
        -------
        List[plt.Axes]
            Returns a 1D list of Axes objects representing the subplot grid.

        Notes
        -----
        This method plots the average distance maps for selected ensembles, where each distance map
        represents the average pairwise distances between residues in a protein structure.
        """

        ens_dict = self._get_distance_matrix_ens_dict()
        num_proteins = len(ens_dict)
        cols = 2  # Number of columns for subplots
        rows = (num_proteins + cols - 1) // cols  # Calculate number of rows needed

        if ax is None:
            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=dpi)
            axes = axes.flatten()  # Ensure axes is a 1D array
        else:
            ax_array = np.array(ax).flatten()
            axes = ax_array  # If ax is provided, flatten it to 1D
            fig = axes[0].figure

        for i, (protein_name, ens_data) in enumerate(ens_dict.items()):
            ax = axes[i]
            
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

            im.set_clim(0, np.ceil(np.nanmax(avg_dmap.flatten())))  # Find the maximum distance and round it to the next integer to manage the auto range
        
        # Remove any empty subplots
        for i in range(num_proteins, rows * cols):
            fig.delaxes(axes[i])

        fig.tight_layout()

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'avg_dmap_' + self.analysis.ens_codes[0]))

        return axes    

    def end_to_end_distances(self, rg_norm: bool = False, bins: int = 50, hist_range: Tuple = None, violin_plot: bool = True, means: bool = True, median: bool = True, save: bool = False, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot end-to-end distance distributions.

        Parameters
        ----------
        rg_norm: bool, optional
            Normalize end-to-end distances on the average radius of gyration.
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        hist_range: Tuple, optional
            A tuple with a min and max value for the histogram. Default is None,
            which corresponds to using the min a max value across all data.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is True.
        means : bool, optional
            If True, means are shown in the violin plot. Default is True.
        median : bool, optional
            If True, medians are shown in the violin plot. Default is True.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : plt.Axes, optional
            The axes on which to plot. Default is None, which creates a new figure and axes.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        ensembles = self.analysis.ensembles

        # Calculate features.
        hist_data = []
        labels = []

        for ensemble in ensembles:
            ca_indices = ensemble.trajectory.topology.select(ensemble.atom_selector)
            hist_data_i = mdtraj.compute_distances(
                ensemble.trajectory, [[ca_indices[0], ca_indices[-1]]]
            ).ravel()
            if rg_norm:
                rg_i = mdtraj.compute_rg(ensemble.trajectory).mean()
                hist_data_i = hist_data_i / rg_i
            hist_data.append(hist_data_i)
            labels.append(ensemble.code)

        # Plot.
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if not rg_norm:
            axis_label = "End-to-End distance [nm]"
            title = "End-to-End distances distribution"
        else:
            axis_label = r"End-to-End distance over $\langle$R$_g$$\rangle$"
            title = r"End-to-End distance over $\langle$R$_g$$\rangle$ distribution"

        if violin_plot:
            plot_violins(
                ax=ax,
                data=hist_data,
                labels=labels,
                means=means,
                median=median,
                title=title,
                xlabel=axis_label
            )
        else:
            plot_histogram(
                ax=ax,
                data=hist_data,
                labels=labels,
                bins=bins,
                range=hist_range,
                title=title,
                xlabel=axis_label
            )

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'e2e_distances_' + self.analysis.ens_codes[0]))

        return ax

    def asphericity(self, bins: int = 50, hist_range: Tuple = None, violin_plot: bool = True, means: bool = True, median: bool = True, save: bool = False, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot asphericity distribution in each ensemble.
        Asphericity is calculated based on the gyration tensor.

        Parameters
        ----------
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        hist_range: Tuple, optional
            A tuple with a min and max value for the histogram. Default is None,
            which corresponds to using the min a max value across all data.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is True.
        means : bool, optional
            If True, means are shown in the violin plot. Default is True.
        median : bool, optional
            If True, medians are shown in the violin plot. Default is True.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : plt.Axes, optional
            The axes on which to plot. Default is None, which creates a new figure and axes.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        ensembles = self.analysis.ensembles

        # Calculate features.
        asph_list = []
        labels = []
        for ensemble in ensembles:
            asphericity = compute_asphericity(ensemble.trajectory)
            asph_list.append(asphericity)
            labels.append(ensemble.code)

        # Plot.
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        axis_label = "Asphericity"
        title = "Asphericity distribution"

        if violin_plot:
            plot_violins(
                ax=ax,
                data=asph_list,
                labels=labels,
                means=means,
                median=median,
                title=title,
                xlabel=axis_label
            )
        else:
            plot_histogram(
                ax=ax,
                data=asph_list,
                labels=labels,
                bins=bins,
                range=hist_range,
                title=title,
                xlabel=axis_label
            )

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'asphericity_dist_' + self.analysis.ens_codes[0]))

        return ax

    def prolateness(self, bins: int = 50, hist_range: Tuple = None, violin_plot: bool = True, median: bool = False, means: bool = False, save: bool = False, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot prolateness distribution in each ensemble.
        Prolateness is calculated based on the gyration tensor.

        Parameters
        ----------
        bins : int, optional
            The number of bins for the histogram. Default is 50.
        hist_range : Tuple, optional
            A tuple with a min and max value for the histogram. Default is None,
            which corresponds to using the min a max value across all data.
        violin_plot : bool, optional
            If True, a violin plot is visualized. Default is True.
        median : bool, optional
            If True, median is showing in the violin plot. Default is False.
        means : bool, optional
            If True, mean is showing in the violin plot. Default is False.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : plt.Axes, optional
            The axes on which to plot. Default is None, which creates a new figure and axes.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        ensembles = self.analysis.ensembles

        # Calculate features.
        prolat_list = []
        labels = []
        for ensemble in ensembles:
            prolat = compute_prolateness(ensemble.trajectory)
            prolat_list.append(prolat)
            labels.append(ensemble.code)

        # Plot.
        if ax is None:
            fig, ax = plt.subplots()  # Create a new figure if ax is not provided
        else:
            fig = ax.figure  # Use the figure associated with the provided ax

        axis_label = "Prolateness"
        title = "Prolateness distribution"

        if violin_plot:
            plot_violins(
                ax=ax,
                data=prolat_list,
                labels=labels,
                means=means,
                median=median,
                title=title,
                xlabel=axis_label
            )
        else:
            plot_histogram(
                ax=ax,
                data=prolat_list,
                labels=labels,
                bins=bins,
                range=hist_range,
                title=title,
                xlabel=axis_label
            )

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'prolateness_dist_' + self.analysis.ens_codes[0]))

        return ax

    def alpha_angles(self, bins: int = 50, save: bool = False, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the distribution of alpha angles.

        Parameters
        ----------
        bins : int
            The number of bins for the histogram. Default is 50.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : plt.Axes, optional
            The axes on which to plot. Default is None, which creates a new figure and axes.

        Returns
        -------
        plt.Axes
            The Axes object containing the plot.
        """

        ensembles = self.analysis.ensembles

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        data = []
        labels = []
        for ensemble in ensembles:
            data_i = featurize_a_angle(
                ensemble.trajectory,
                get_names=False,
                atom_selector=ensemble.atom_selector
            ).ravel()
            data.append(data_i)
            labels.append(ensemble.code)

        plot_histogram(
            ax=ax,
            data=data,
            labels=labels,
            bins=bins,
            range=(-np.pi, np.pi),
            title="Distribution of alpha angles",
            xlabel="angle [rad]"
        )

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'alpha_dist_' + self.analysis.ens_codes[0]))

        return ax

    def contact_prob_maps(self,
                        norm=True, 
                        min_sep=2,
                        max_sep=None,
                        threshold: float = 0.8,
                        dpi: int = 96, 
                        save: bool = False, 
                        cmap_color='Blues',
                        ax: Union[None, List[plt.Axes], np.ndarray] = None) -> Union[List[plt.Axes], np.ndarray]:
        from matplotlib.colors import LogNorm
        """
        Plot the contact probability map based on the threshold.

        Parameters
        ----------
        norm : bool, optional
            If True, use log scale range. Default is True.
        min_sep : int, optional
            Minimum separation distance between atoms to consider. Default is 2.
        max_sep : int, optional
            Maximum separation distance between atoms to consider. Default is None.
        threshold : float, optional
            Determining the threshold for calculating the contact frequencies. Default is 0.8 [nm].
        dpi : int, optional
            For changing the quality and dimension of the output figure. Default is 96.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        cmap_color : str, optional
            Select a color for the contact map. Default is "Blues".

        Returns
        -------
        Union[List[plt.Axes], np.ndarray]
            Returns a list or array of Axes objects representing the subplot grid.
        """

        ensembles = self.analysis.ensembles
        num_proteins = len(ensembles)
        num_cols = 2
        num_rows = (num_proteins + num_cols - 1) // num_cols

        cmap = plt.get_cmap(cmap_color)
        
        if ax is None:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows), dpi=dpi)
            axes = axes.flatten()
        else:
            ax_array = np.array(ax)
            axes = ax_array.flatten()
            fig = axes[0].figure

        for i, ensemble in enumerate(ensembles):
            ax = axes[i]
            
            if ensemble.coarse_grained:
                matrix_p_map = contact_probability_map(ensemble.trajectory, scheme='closest', contact=self._pair_ids(min_sep=min_sep, max_sep=max_sep), threshold=threshold)
            else:
                matrix_p_map = contact_probability_map(ensemble.trajectory, threshold=threshold)

            if norm:
                im = ax.imshow(matrix_p_map, cmap=cmap, norm=LogNorm())
            else:
                im = ax.imshow(matrix_p_map, cmap=cmap)
            ax.set_title(f"Contact Probability Map: {ensemble.code}", fontsize=14)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Frequency', fontsize=14)
            cbar.ax.tick_params(labelsize=14)

        # Remove any empty subplots
        for i in range(num_proteins, num_rows * num_cols):
            fig.delaxes(axes[i])
        
        fig.tight_layout()

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'contact_prob_' + self.analysis.ens_codes[0]))

        return axes

    def _pair_ids(self, min_sep=2,max_sep = None ):
        analysis = self.analysis
        pair_ids = []
        for ens in analysis.ensembles:
            ca_ids = ens.trajectory.topology.select('name')
            atoms = list(ens.trajectory.topology.atoms)
            max_sep = get_max_sep(L=len(atoms), max_sep=max_sep)
    # Get all pair of ids.
            for i, id_i in enumerate(ca_ids):
                for j, id_j in enumerate(ca_ids):
                    if j - i >= min_sep:
                        if j - i > max_sep:
                            continue
                        pair_ids.append([id_i, id_j])
        return pair_ids
    
    def ramachandran_plots(
            self,
            two_d_hist: bool = True,
            linespaces: Tuple = (-180, 180, 80),
            save: bool = False,
            ax: Union[None, plt.Axes, np.ndarray, List[plt.Axes]] = None
    ) -> Union[List[plt.Axes], plt.Axes]:
        """
        Ramachandran plot. If two_d_hist=True it returns a 2D histogram 
        for each ensemble. If two_d_hist=False it returns a simple scatter plot 
        for all ensembles in one plot.

        Parameters
        ----------
        two_d_hist : bool, optional
            If True, it returns a 2D histogram for each ensemble. Default is True.
        linespaces : tuple, optional
            You can customize the bins for 2D histogram. Default is (-180, 180, 80).
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : Union[None, plt.Axes, np.ndarray, List[plt.Axes]], optional
            The axes on which to plot. If None, new axes will be created. Default is None.

        Returns
        -------
        Union[List[plt.Axes], plt.Axes]
            If two_d_hist=True, returns a list of Axes objects representing the subplot grid for each ensemble. 
            If two_d_hist=False, returns a single Axes object representing the scatter plot for all ensembles.

        """

        if self.analysis.exists_coarse_grained():
            raise ValueError("This analysis is not possible with coarse-grained models.")
        
        ensembles = self.analysis.ensembles
        if two_d_hist:
            if ax is None:
                fig, ax = plt.subplots(1, len(ensembles), figsize=(5 * len(ensembles), 5))
            else:
                if not isinstance(ax, (list, np.ndarray)):
                    ax = [ax]
                ax = np.array(ax).flatten()
                fig = ax[0].figure
            # Ensure ax is always a list
            if not isinstance(ax, np.ndarray):
                ax = [ax]
            rama_linspace = np.linspace(linespaces[0], linespaces[1], linespaces[2])
            for ens, axis in zip(ensembles, ax):
                phi_flat = np.degrees(mdtraj.compute_phi(ens.trajectory)[1]).ravel()
                psi_flat = np.degrees(mdtraj.compute_psi(ens.trajectory)[1]).ravel()
                hist = axis.hist2d(
                    phi_flat,
                    psi_flat,
                    cmap="viridis",
                    bins=(rama_linspace, rama_linspace), 
                    norm=colors.LogNorm(),
                    density=True
                )

                axis.set_title(f'Ramachandran Plot for cluster {ens.code}')
                axis.set_xlabel('Phi (ϕ) Angle (degrees)')
                axis.set_ylabel('Psi (ψ) Angle (degrees)')
            fig.tight_layout()
        else:
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                fig = ax.figure
            for ens in ensembles:
                phi = np.degrees(mdtraj.compute_phi(ens.trajectory)[1])
                psi = np.degrees(mdtraj.compute_psi(ens.trajectory)[1])
                ax.scatter(phi, psi, s=1, label=ens.code)
            ax.set_xlabel('Phi (ϕ) Angle (degrees)')
            ax.set_ylabel('Psi (ψ) Angle (degrees)')
            ax.legend()

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'ramachandran_' + self.analysis.ens_codes[0]))  

        return ax

    def ss_flexibility_parameter(self, 
                                pointer: List[int] = None, 
                                figsize: Tuple[int, int] = (15, 5), 
                                save: bool = False,
                                ax: Union[None, plt.Axes] = None) -> plt.Axes:
        """
        Generate a plot of the site-specific flexibility parameter.
        
        This plot shows the site-specific measure of disorder, which is sensitive to local flexibility based on 
        the circular variance of the Ramachandran angles φ and ψ for each residue in the ensemble.
        The score ranges from 0 for identical dihedral angles for all conformers at the residue i to 1 for a 
        uniform distribution of dihedral angles at the residue i.
        
        Parameters
        ----------
        pointer: List[int], optional
            A list of desired residues. Vertical dashed lines will be added to point to these residues. Default is None.
        figsize: Tuple[int, int], optional
            The size of the figure. Default is (15, 5).
        save : bool, optional
            If True, save the plot as an image file. Default is False.
        ax : Union[None, plt.Axes], optional
            The matplotlib Axes object on which to plot. If None, a new Axes object will be created. Default is None.
            
        Returns
        -------
        plt.Axes
            The matplotlib Axes object containing the plot.
        """
        
        if self.analysis.exists_coarse_grained():
            raise ValueError("This analysis is not possible with coarse-grained models.")
        
        features_dict = self.analysis.get_features(featurization='phi_psi')
        
        f = ss_measure_disorder(features_dict)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        for key, values in f.items():
            x = np.arange(1, len(values) + 1)
            ax.plot(x, values, marker='o', linestyle='-', label=key)
        
        ax.set_xticks([i for i in x if i == 1 or i % 5 == 0])
        ax.set_xlabel("Residue Index")
        ax.set_ylabel("Site-specific flexibility parameter")
        ax.legend()
        
        if pointer is not None:
            for res in pointer:
                ax.axvline(x=res, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'ss_flexibility_' + self.analysis.ens_codes[0]))  

        return ax

    def ss_order_parameter(self, 
                        pointer: List[int] = None, 
                        figsize: Tuple[int, int] = (15, 5), 
                        save: bool = False, 
                        ax: Union[None, plt.Axes] = None) -> plt.Axes:
        """
        Generate a plot of the site-specific order parameter.
        
        This plot shows the site-specific order parameter, which abstracts from local chain flexibility.
        The parameter is still site-specific, as orientation correlations in IDRs and IDPs decrease with increasing sequence distance.
        
        Parameters
        ----------
        pointer: List[int], optional
            A list of desired residues. Vertical dashed lines will be added to point to these residues. Default is None.
        figsize: Tuple[int, int], optional
            The size of the figure. Default is (15, 5).
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : Union[None, plt.Axes], optional
            The matplotlib Axes object on which to plot. If None, a new Axes object will be created. Default is None.
            
        Returns
        -------
        plt.Axes
            The matplotlib Axes object containing the plot.
        """
        
        ensembles = self.analysis.ensembles
        dict_ca_xyz = {}
        for ensemble in ensembles:
            ca_index = ensemble.trajectory.topology.select(ensemble.atom_selector)
            dict_ca_xyz[ensemble.code] = ensemble.trajectory.xyz[:, ca_index, :]

        dict_order_parameter = site_specific_order_parameter(dict_ca_xyz)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        for key, values in dict_order_parameter.items():
            x = np.arange(1, len(values) + 1)
            ax.plot(x, values, label=key, marker= 'o', linestyle='-')
        
        ax.set_xticks([i for i in x if i == 1 or i % 5 == 0])
        ax.set_xlabel("Residue Index")
        ax.set_ylabel("Site-specific order parameter")
        ax.legend()
        
        if pointer is not None:
            for res in pointer:
                ax.axvline(x=res, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        
        if save:
            fig.savefig(os.path.join(self.plot_dir, 'ss_order_' + self.analysis.ens_codes[0]))  
        
        return ax

    def per_residue_mean_sasa(self, 
                            figsize: Tuple[int, int] = (15, 5), 
                            pointer: List[int] = None, 
                            save: bool = False, 
                            ax: Union[None, plt.Axes] = None) -> plt.Axes:
        """
        Plot the average solvent-accessible surface area (SASA) for each residue among all conformations in an ensemble.

        Parameters
        ----------
        figsize: Tuple[int, int], optional
            Tuple specifying the size of the figure. Default is (15, 5).
        pointer: List[int], optional
            List of desired residues to highlight with vertical dashed lines. Default is None.
        save : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        ax : Union[None, plt.Axes], optional
            The matplotlib Axes object on which to plot. If None, a new Axes object will be created. Default is None.

        Returns
        -------
        plt.Axes
            Axes object containing the plot.

        """

        analysis = self.analysis

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        # Get the color cycle from matplotlib
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        for i, ens in enumerate(analysis.ensembles):
            color = colors[i % len(colors)]
            res_based_sasa = mdtraj.shrake_rupley(ens.trajectory, mode='residue')
            sasa_mean = np.mean(res_based_sasa, axis=0)
            sasa_std = np.std(res_based_sasa, axis=0)        

            ax.plot(np.arange(1, len(sasa_mean) + 1), sasa_mean, '-o', color=color, label=ens.code)
            # ax.fill_between(np.arange(1, len(sasa_mean) + 1), sasa_mean - sasa_std, sasa_mean + sasa_std, alpha=0.3, color=colors[i % len(colors)])
            ax.plot(np.arange(1, len(sasa_mean) + 1), sasa_mean + sasa_std, '--', color=color, alpha=0.5)
            ax.plot(np.arange(1, len(sasa_mean) + 1), sasa_mean - sasa_std, '--', color=color, alpha=0.5)

        ax.set_xticks([i for i in np.arange(1, len(sasa_mean) + 1) if i == 1 or i % 5 == 0])
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Mean SASA')
        ax.set_title('Mean SASA for Each Residue in Ensembles')
        ax.legend()
        # ax.grid(True)
        
        if pointer is not None:
            for res in pointer:
                ax.axvline(x=res, color='blue', linestyle='--', alpha=0.3, linewidth=1)

        if save:
            fig.savefig(os.path.join(self.plot_dir, 'local_sasa_' + self.analysis.ens_codes[0]))  

        return ax

    def ca_com_distances(self, 
                         min_sep: int = 2, 
                         max_sep: Union[int, None] = None, 
                         get_names: bool = True, 
                         inverse: bool = False,
                         save: bool = False,
                         ax: Union[None, plt.Axes, np.ndarray, List[plt.Axes]] = None
                        ) -> List[plt.Axes]:
        """
        Plot the distance maps comparing the center of mass (COM) and alpha-carbon (CA) distances within each ensemble.

        Parameters:
        -----------
        min_sep : int, optional
            Minimum separation distance between atoms to consider. Default is 2.
        max_sep : int or None, optional
            Maximum separation distance between atoms to consider. Default is None, which means no maximum separation.
        get_names : bool, optional
            Whether to get the residue names for the features. Default is True.
        inverse : bool, optional
            Whether to compute the inverse distances. Default is False.
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (6, 2.5).
        save : bool, optional
            If True, save the plot as an image file. Default is False.

        Returns:
        --------
        List[plt.Axes]
            A list containing Axes objects corresponding to the plots for CA and COM distances.

        Notes:
        ------
        This method plots the average distance maps for the center of mass (COM) and alpha-carbon (CA) distances
        within each ensemble. It computes the distance matrices for COM and CA atoms and then calculates their
        mean values to generate the distance maps. The plots include color bars indicating the distance range.
        """

        if self.analysis.exists_coarse_grained():
            raise ValueError("This analysis is not possible with coarse-grained models.")
        
        num_proteins = len(self.analysis.ensembles)
        
        if ax is None:
            fig, axes = plt.subplots(2, num_proteins, figsize=(10, 4 * num_proteins))
            axes = axes.flatten()
        else:
            ax_array = np.array(ax)
            axes = ax_array.flatten()
            fig = axes[0].figure

        for i, ens in enumerate(self.analysis.ensembles):
            idx = i * 2
            traj = ens.trajectory
            feat, names = featurize_com_dist(traj=traj, min_sep=min_sep,max_sep=max_sep,inverse=inverse ,get_names=get_names)  # Compute (N, *) feature arrays.
            print(f"# Ensemble: {ens.code}")
            print("features:", feat.shape)

            com_dmap = calc_ca_dmap(traj=traj)
            com_dmap_mean = com_dmap.mean(axis=0)
            ca_dmap = calc_ca_dmap(traj=traj)
            ca_dmap_mean = ca_dmap.mean(axis=0)

            print("distance matrix:", com_dmap_mean.shape)
            
            im0 = axes[idx].imshow(ca_dmap_mean)
            axes[idx].set_title(f"{ens.code} CA")
            im1 = axes[idx + 1].imshow(com_dmap_mean)
            axes[idx + 1].set_title(f"{ens.code} COM")
            cbar = fig.colorbar(im0, ax=axes[idx], shrink=0.8)
            cbar.set_label("distance [nm]")
            cbar = fig.colorbar(im1, ax=axes[idx + 1], shrink=0.8)
            cbar.set_label("distance [nm]")

            fig.tight_layout()

            if save:
                fig.savefig(os.path.join(self.plot_dir, 'dist_ca_com_' + ens.code))  

        return axes
    
    
    def comparison_matrix(self,
            score: str,
            feature: str,
            featurization_params: dict = {},
            bootstrap_iters: int = 3,
            bootstrap_frac: float = 1.0,
            bootstrap_replace: bool = True,
            bins: Union[int, str] = 50,
            random_seed: int = None,
            verbose: bool = False,
            ax: Union[None, plt.Axes] = None,
            figsize: Tuple[int] = (6.00, 5.0),
            dpi: int = 100,
            std: bool = False,
            cmap: str = "viridis_r",
            title: str = None,
            cbar_label: str = None,
            textcolors: Union[str, tuple] = ("black", "white")
        ) -> dict:
        """
        Generates and visualizes the pairwise comparison matrix for the ensembles.
        This function computes the comparison matrix using the specified score
        type and feature. It then visualizes the matrix using a heatmap.

        Parameters:
        -----------
        score, feature, featurization_params, bootstrap_iters, bootstrap_frac,
        bootstrap_replace, bins, random_seed, verbose:
            See the documentation of `EnsembleAnalysis.comparison_scores` for
            more information about these arguments.
        ax: Union[None, plt.Axes], optional
            Axes object where to plot the comparison heatmap. If `None` (the
            default value) is provided, a new Figure will be created.
        figsize: Tuple[int], optional
            The size of the figure for the heatmap. Default is (6.00, 5.0). Only
            takes effect if `ax` is not `None`.
        dpi: int, optional
            DPIs of the figure for the heatmap. Default is 100. Only takes
            effect if `ax` is not `None`.
        std, cmap, title, cbar_label, textcolors:
            See the documentation of `dpet.visualization.plot_comparison_matrix`
            for more information about these arguments.

        Returns:
        --------
        results: dict
            A dictionary containing the following keys:
                `ax`: the Axes object with the comparison matrix heatmap.
                `scores`: comparison matrix. See `EnsembleAnalysis.comparison_scores`
                    for more information.
                `codes`: codes of the ensembles that were compared.
                `fig`: Figure object, only returned when a new figure is created
                    inside this function.

        Notes:
        ------
        The comparison matrix is annotated with the scores, and the axes are
        labeled with the ensemble labels.

        """

        ### Score divergences.
        scores, codes = self.analysis.comparison_scores(
            score=score,
            feature=feature,
            featurization_params=featurization_params,
            bootstrap_iters=bootstrap_iters,
            bootstrap_frac=bootstrap_frac,
            bootstrap_replace=bootstrap_replace,
            bins=bins,
            random_seed=random_seed,
            verbose=verbose
        )

        ### Setup the plot.
        # Axes.
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = None
        # Title.
        if title is None:
            if score == "emd":
                title = f"{score.upper()} based on {feature}"
            elif score == "jsd":
                if feature == "ca_dist":
                    title = "aJSD_d"
                elif feature == "alpha_angle":
                    title = "aJSD_t"
                else:
                    title = f"{score.upper()} based on {feature}"
            else:
                raise ValueError(score)
        # Colorbar label.
        if cbar_label is None:
            cbar_label = f"{score.upper()} score"

        ### Actually plots.
        plot_comparison_matrix(
            ax=ax,
            scores=scores,
            codes=codes,
            std=std,
            cmap=cmap,
            title=title,
            cbar_label=cbar_label,
            textcolors=textcolors
        )

        ### Return results.
        results = {"ax": ax, "scores": scores, "codes": codes}
        if fig is not None:
            results["fig"] = fig
        return results
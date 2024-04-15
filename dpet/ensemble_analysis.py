from pathlib import Path
import re
import shutil
import zipfile
from dpet.api_client import APIClient
from dpet.featurization.angles import featurize_a_angle, featurize_phi_psi, featurize_tr_angle
from dpet.featurization.distances import featurize_ca_dist
from dpet.visualization.reports import generate_custom_report, generate_dimenfix_report, generate_pca_report, generate_tsne_report
import dpet.visualization.visualization as visualization
from dpet.utils import extract_tar_gz
import os
import mdtraj
import numpy as np
from dpet.dimensionality_reduction.dimensionality_reduction import DimensionalityReductionFactory

PLOT_DIR = "plots"

class EnsembleAnalysis:
    def __init__(self, ens_codes, data_dir: str):
        self.data_dir = Path(data_dir)
        self.api_client = APIClient()
        self.trajectories = {}
        self.feature_names = []
        self.featurized_data = {}
        self.all_labels = []
        self.ens_codes = ens_codes
        plot_dir = os.path.join(self.data_dir, PLOT_DIR)
        os.makedirs(plot_dir, exist_ok=True)
        self.figures = {}

    def __del__(self):
        if hasattr(self, 'api_client'):
            self.api_client.close_session()
    
    def download_from_ped(self):
        """Automate Downloading ensembles
        using PED API 
        """
        # Define the pattern
        pattern = r'^(PED\d+)(e\d+)$'

        # Filter the ens_codes list using regex
        for ens_code in self.ens_codes:
            match = re.match(pattern, ens_code)
            if match:
                print(f"Downloading entry {ens_code} from PED.")
                ped_id = match.group(1)
                ensemble_id = match.group(2)
                tar_gz_filename = f'{ens_code}.tar.gz'
                tar_gz_file = os.path.join(self.data_dir, tar_gz_filename)

                pdb_filename = f'{ens_code}.pdb'
                pdb_file = os.path.join(self.data_dir, pdb_filename)

                if not os.path.exists(tar_gz_file) and not os.path.exists(pdb_file):
                    url = f'https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb'
                    print(url)
                    headers = {'accept': '*/*'}

                    response = self.api_client.perform_get_request(url, headers=headers)
                    if response:
                        # Download and save the response content to a file
                        self.api_client.download_response_content(response, tar_gz_file)
                        print(f"Downloaded file {tar_gz_filename} from PED.")
                else:
                    print("File already exists. Skipping download.")

                # Extract the .tar.gz file
                if not os.path.exists(pdb_file):
                    extract_tar_gz(tar_gz_file, self.data_dir, pdb_filename)
                    print(f"Extracted file {pdb_filename}.")
                else:
                    print("File already exists. Skipping extracting.")
            else:
                print(f"Entry {ens_code} does not match the pattern and will be skipped.")

    
    def download_from_atlas(self):
        """ Automate Downloading MD ensembles from
        Atlas. 
        """
        new_ens_codes = []
        for ens_code in self.ens_codes:
            print(f"Downloading entry {ens_code} from Atlas.")
            zip_filename = f'{ens_code}.zip'
            zip_file = os.path.join(self.data_dir, zip_filename)

            url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{ens_code}/{ens_code}_protein.zip"
            headers = {'accept': '*/*'}

            response = self.api_client.perform_get_request(url, headers=headers)
            if response:
                # Download and save the response content to a file
                self.api_client.download_response_content(response, zip_file)
                print(f"Downloaded file {zip_filename} from Atlas.")
                # Unzip.
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                    print(f"Extracted directory {self.data_dir}.")

                # Remove unused files.
                for unused_path in self.data_dir.glob("*.tpr"):
                    os.remove(unused_path)
                os.remove(self.data_dir / "README.txt")
                os.remove(zip_file)

        # Collect xtc files for updating ens_codes
        new_ens_codes = ([f.stem for f in self.data_dir.glob("*.xtc")])

        # Copy and rename toplogy files to new ensemble codes
        for ens_code in self.ens_codes:
            pdb_file = os.path.join(self.data_dir,f"{ens_code}.pdb")
            for new_ens_code in new_ens_codes:
                if new_ens_code.__contains__(ens_code):
                    new_pdb_file = os.path.join(self.data_dir,f"{new_ens_code}.top.pdb")
                    shutil.copy(pdb_file, new_pdb_file)
                    print(f"Copied and renamed {pdb_file} to {new_pdb_file}.")
            # Delete old topology file
            os.remove(pdb_file)

        # Update self.ens_codes
        self.ens_codes = new_ens_codes
        print("Analysing ensembles:", self.ens_codes)

    def download_from_database(self, database: str =None):
        """ Download ensembles from databases

        Parameter
        ---------
        databse : str
        Choose the database you want to download from ('ped'/'atlas')
        """
        if database == "ped":
            self.download_from_ped()
        elif database == "atlas":
            self.download_from_atlas()

    def generate_trajectories(self):

        """
        Loading trajectory files on to mdtraj object.
        if only pdb files are existed, the function makes dcd and 
        topology files for the fast loading next times.
        """
        for ens_code in self.ens_codes:
            pdb_filename = f'{ens_code}.pdb'
            pdb_file = os.path.join(self.data_dir, pdb_filename)
            traj_dcd = os.path.join(self.data_dir, f'{ens_code}.dcd')
            traj_xtc = os.path.join(self.data_dir, f'{ens_code}.xtc')
            traj_top = os.path.join(self.data_dir, f'{ens_code}.top.pdb')
            
            ens_dir = os.path.join(self.data_dir, ens_code)

            if os.path.exists(traj_dcd) and os.path.exists(traj_top):
                print(f'Trajectory already exists for ensemble {ens_code}. Loading trajectory.')
                trajectory = mdtraj.load(traj_dcd, top=traj_top)
                self.trajectories[ens_code] = trajectory
            elif os.path.exists(traj_xtc) and os.path.exists(traj_top):
                print(f'Trajectory already exists for ensemble {ens_code}. Loading trajectory.')
                trajectory = mdtraj.load(traj_xtc, top=traj_top)
                self.trajectories[ens_code] = trajectory
            elif os.path.exists(pdb_file):
                print(f'Generating trajectory from PDB file: {pdb_file}.')
                trajectory = mdtraj.load(pdb_file)
                print(f'Saving trajectory.')
                trajectory.save(traj_dcd)
                trajectory[0].save(traj_top)
                self.trajectories[ens_code] = trajectory
            elif os.path.exists(ens_dir):
                files_in_dir = [f for f in os.listdir(ens_dir) if f.endswith('.pdb')]
                if files_in_dir:
                    full_paths = [os.path.join(ens_dir, file) for file in files_in_dir]
                    print(f'Generating trajectory from directory: {ens_dir}.')
                    trajectory = mdtraj.load(full_paths)
                    print(f'Saving trajectory.')
                    trajectory.save(traj_dcd)
                    trajectory[0].save(traj_top)
                    self.trajectories[ens_code] = trajectory
                else:
                    print(f"No DCD files found in directory: {ens_dir}")
            else:
                print(f"File or directory for ensemble {ens_code} doesn't exist.")
                return
            # Copy in order to be able to sample multiple times
            self.old_trajectories = self.trajectories.copy()
            
    def random_sample_trajectories(self, sample_size):
        """
        Sample a defined random number of conformations from the ensemble 
        trajectroy. 

        Note: If you want to change the sample size after running the cell, you need to load 
        the trajectories again. 

        Parameters
        ----------
        sample_size: int
        Number of conformations sampled from the ensemble. 
        """
        self.trajectories = {ensemble_id: self._random_sample(traj, sample_size) for ensemble_id, traj in self.old_trajectories.items()}

    def _random_sample(self, trajectory, sample_size):
        total_frames = len(trajectory)
        random_indices = np.random.choice(total_frames, size=sample_size, replace=False)
        subsampled_traj = mdtraj.Trajectory(
            xyz=trajectory.xyz[random_indices],
            topology=trajectory.topology)
        return subsampled_traj

    def perform_feature_extraction(self, featurization: str, normalize = False, *args, **kwargs):
        """
        Extract the selected feature. The options are "phi_psi", "ca_dist" and "a_angle".

        Note: If you want to change the extracted feature after running this method you need to restart the kernel. 
        
        Parameters
        ----------
        featurization: str 
        Choose between "phi_psi", "ca_dist" and "a_angle"

        normalize: Bool
        if featurization is "ca_dist" normalize True will normalize the distances based on the mean and standard deviation.

        """
        self._extract_features(featurization, *args, **kwargs)
        self.concat_features = self._get_concat_features()
        self._create_all_labels()
        if normalize and featurization == "ca_dist":
            self._normalize_data()

    def _extract_features(self, featurization: str, *args, **kwargs):
        get_names = True
        self.featurization = featurization
        for ens_code, trajectory in self.trajectories.items():
            print(f"Performing feature extraction for Ensemble: {ens_code}.")
            if get_names:
                features, names = self._featurize(featurization, trajectory, get_names, *args, **kwargs)
                get_names = False
            else:
                features = self._featurize(featurization, trajectory, get_names, *args, **kwargs)
            self.featurized_data[ens_code] = features
            print("Transformed ensemble shape:", features.shape)
        self.feature_names = names
        print("Feature names:", names)

    def _featurize(self, featurization, trajectory, get_names, *args, **kwargs):
        if featurization == "ca_dist":
            return featurize_ca_dist(traj=trajectory, get_names=get_names, *args, **kwargs)
        elif featurization == "phi_psi":
            return featurize_phi_psi(traj=trajectory, get_names=get_names, *args, **kwargs)
        elif featurization == "a_angle":
            return featurize_a_angle(traj=trajectory, get_names=get_names, *args, **kwargs)
        elif featurization == "tr_omega":
            return featurize_tr_angle(
                traj=trajectory,
                type="omega",
                get_names=get_names,
                *args, **kwargs)
        elif featurization == "tr_phi":
            return featurize_tr_angle(
                traj=trajectory,
                type="phi",
                get_names=get_names,
                *args, **kwargs)
        else:
            raise NotImplementedError("Unsupported feature extraction method.")

    def _create_all_labels(self):
        self.all_labels = []
        for label, data_points in self.featurized_data.items():
            num_data_points = len(data_points)
            self.all_labels.extend([label] * num_data_points)

    def _normalize_data(self):
        mean = self.concat_features.mean(axis=0)
        std = self.concat_features.std(axis=0)
        self.concat_features = (self.concat_features - mean) / std
        for label, features in self.featurized_data.items():
            self.featurized_data[label] = (features - mean) / std

    def _calculate_rg_for_trajectory(self, trajectory):
        return [mdtraj.compute_rg(frame) for frame in trajectory]

    def rg_calculator(self):
        """
        Calculates Rg for each conformations in the loaded ensembles.
        The returned values are in Angstrom.  
        """
        rg_values_list = []
        for traj in self.trajectories.values():
            rg_values_list.extend(self._calculate_rg_for_trajectory(traj))
            self.rg = [item[0] * 10 for item in rg_values_list]
        return self.rg

    def _get_concat_features(self, fit_on:list=None):
        if fit_on is None:
            fit_on = self.ens_codes
        
        concat_features = [features for ens_id, features in self.featurized_data.items() if ens_id in fit_on]
        concat_features = np.concatenate(concat_features, axis=0)
        print("Concatenated featurized ensemble shape:", concat_features.shape)
        return concat_features

    def fit_dimensionality_reduction(self, method: str, fit_on: list=None, *args, **kwargs):
        self.reducer = DimensionalityReductionFactory.get_reducer(method, *args, **kwargs)
        self.reduce_dim_method = method
        if method in ("pca","kpca"):
            fit_on_data = self._get_concat_features(fit_on=fit_on)
            self.reduce_dim_model = self.reducer.fit(data=fit_on_data)
            self.reduce_dim_data = {}
            for key, data in self.featurized_data.items():
                self.reduce_dim_data[key] = self.reducer.transform(data)
                print("Reduced dimensionality ensemble shape:", self.reduce_dim_data[key].shape)
            self.transformed_data = self.reducer.transform(data=self.concat_features)
        else:
            self.transformed_data = self.reducer.fit_transform(data=self.concat_features)

    def execute_pipeline(self, featurization_params:dict, reduce_dim_params:dict, database:str=None, subsample_size:int=None) -> None:
        self.download_from_database(database)
        self.generate_trajectories()
        if subsample_size is not None:
            self.random_sample_trajectories(subsample_size)
        self.perform_feature_extraction(**featurization_params)
        self.rg_calculator()
        self.fit_dimensionality_reduction(**reduce_dim_params)

    ##################### Integrated plot function #####################



    def tsne_ramachandran_plot_density(self, save=False):
        """
        It gets the 2-D histogram ramachandran plots of the
        clusters from t-SNE analysis. \n
        The results is only meaningful when the extracted feature is "phi_psi".

        Parameters
        -----------
        save: bool
        If True the plot will save 
        """
        # if featurization_option != "phi_psi": (This control step should be added)
            # it should raise an error
        visualization.tsne_ramachandran_plot_density(self, save)

    def tsne_scatter_plot(self, save=False):
        """
        It gets the output results of t-SNE. 
        Three scatter plot will be generated based on original, clustering and Rg labels. 
        One KDE density plot will also be generated to shod the most populated areas in 
        the reduced dimension. 

        

        Parameters
        -----------
        save: Bool \n
        if True the plot will be saved. 
        """

        if self.reduce_dim_method == "tsne":
            visualization.tsne_scatter_plot(self, save)
        else:
            print("Analysis is only valid for t-SNE dimensionality reduction.")

    def tsne_scatter_plot_rg(self, save=False):
        # It plots a redundant plot which implemented in tsne_scatter_plot and could be removed
        visualization.tsne_scatter_plot_rg(self, save)

    def dimenfix_scatter(self, save=False):
        """
        It gets the the final result for dimenfix method. 

        Parameters
        ----------
        save: bool \n
        if True it will save the figure
        """
        visualization.dimenfix_scatter(self, save)

    # def dimenfix_scatter_plot(self, save=False):
    #     visualization.dimenfix_scatter_plot_rg(self, save)

    # def dimenfix_scatter_plot_2(self, save=False):
    #     visualization.dimenfix_scatter_plot_ens(self, save)

    # def dimenfix_cluster_scatter_plot(self, save=False):
    #     visualization.dimenfix_cluster_scatter_plot(self, save)

    # def dimenfix_cluster_scatter_plot_2(self, save=False):
    #     visualization.dimenfix_cluster_scatter_plot_2(self, save)

    def pca_cumulative_explained_variance(self, save=False):

        """
        It gets the cumulative variance and only applicable when the
        dimensionality reduction method is "pca"

        Parameters
        ----------
        save: bool \n
        if True it will save the figure
        """
        if self.reduce_dim_method == "pca":
            visualization.pca_cumulative_explained_variance(self, save)
        else:
            print("Analysis is only valid for PCA dimensionality reduction.")

    def pca_plot_2d_landscapes(self, save=False):

        """
        It gets the 2d landscapes plots when the dimensionality reduction method 
        is "pca" or "kpca"

        Parameters
        ----------
        save: bool \n
        if True it will save the figure
        """
        visualization.pca_plot_2d_landscapes(self, save)

    def pca_plot_1d_histograms(self, save=False):
        """
        It gets the 2d landscapes plots when the dimensionality reduction method 
        is "pca" or "kpca"

        Parameters
        ----------
        save: bool \n
        if True it will save the figure
        """
        visualization.pca_plot_1d_histograms(self, save)

    def pca_correlation_plot(self, num_residues, sel_dims, save=False):
        """
        It plots PCA weight in different dimensions for range of inter-residue distances.
        It only meaningful when the extracted feature is "ca_dist" 

        Parameters
        ----------
        num_residues: int 
        Number of residues.

        sel_dims: list
        list of dimensions you want to analyze (start from 0)

        save: bool \n
        if True it will save the figure
        """
        if self.featurization == "ca_dist":
            visualization.pca_correlation_plot(num_residues, sel_dims, self, save)
        else:
            print("Analysis is only valid for ca_dist feature extraction.")
    
    def pca_rg_correlation(self, save=False):
        """
        It demonstrates the correlation between PC dimension 1 and the amount of Rg
        Typically high correlation can be detected here. 

        Parameters
        ----------
        save: bool \n
        if True it will save the figure
        """
        visualization.pca_rg_correlation(self, save)
        
    def plot_global_sasa(self, save=False, showmeans=True ,showmedians=True):

        """
        It plots the distribution of SASA for each conformation 
        within the ensembles 

        Parameters
        ----------
        save: bool \n
        if True it will save the figure

        showmean: bool \n
        if True it will show the mean 

        showmedian: bool \n 
        if True it will show the median
        """
        visualization.plot_global_sasa(self, save, showmeans, showmedians)

    def plot_rg_vs_asphericity(self, save=False):
        """
        It plots the Rg versus Asphericity and gives the pearson correlation coefficient to evaluate 
        the correlation between Rg and Asphericity. 
        """
        visualization.plot_rg_vs_asphericity(self, save)
    '''
    def trajectories_plot_density(self):
        visualization.trajectories_plot_density(self.trajectories)
    '''

    def plot_rg_vs_prolateness(self, save=False):
        """
        It plots the Rg versus Prolateness and gives the pearson correlation coefficient to evaluate 
        the correlation between Rg and Prolateness. 
        """
        visualization.plot_rg_vs_prolateness(self, save)
    '''
    def trajectories_plot_prolateness(self):
        visualization.trajectories_plot_prolateness(self.trajectories)
    '''
    
    def plot_alpha_angle_dihedral(self, bins=50, atom_selector='protein and name CA'):
        visualization.plot_alpha_angle_dihederal(self, bins, atom_selector)

    def plot_relative_helix_content(self):

        """
        Plot the relative helix content in each ensemble for each residue. 
        """
        visualization.plot_relative_helix_content(self.trajectories)

    def trajectories_plot_rg_comparison(self, n_bins=50, bins_range=(1, 4.5), dpi=96):
        """
        Plot the distribution of the Rg whithin each ensemble
        
        Parameter 
        ---------
        n_bins : int 
        bins_range : tuple
        change the Rg scale in x-axis 
        dpi : int
        """
        visualization.trajectories_plot_rg_comparison(self.trajectories, n_bins, bins_range, dpi)

    def plot_average_dmap_comparison(self, 
                                    ticks_fontsize=14,
                                    cbar_fontsize=14,
                                    title_fontsize=14,
                                    dpi=96,
                                    max_d=6.8,
                                    use_ylabel=True):
        
        """Plot the average distance maps for selected ensembles.
        
        Parameters
        ----------
        ticks_fontsize: int
        cbar_fontsize: int
        title_fontsize: int
        dpi: int
        max_d: float
        The maximum amount for distance the default value is 6.8
        use_ylabel: bool
        """
        visualization.plot_average_dmap_comparison(self.trajectories, ticks_fontsize, cbar_fontsize, title_fontsize, dpi, max_d, use_ylabel)

    def plot_cmap_comparison(self,
                            title,
                            ticks_fontsize=14,
                            cbar_fontsize=14,
                            title_fontsize=14,
                            dpi=96,
                            cmap_min=-3.5,
                            use_ylabel=True):
        visualization.plot_cmap_comparison(self.trajectories, title, ticks_fontsize, cbar_fontsize, title_fontsize, dpi, cmap_min, use_ylabel)

    def plot_distance_distribution_multiple(self, dpi = 96):
        visualization.plot_distance_distribution_multiple(self.trajectories, dpi)

    def end_to_end_distances_plot(self, atom_selector ="name CA", bins = 50, violin_plot=True, means=False, median=True ):
        """
        Plot end-to-end distance distributions. 

        Parameters
        ----------
        atom_selector: str 
        The type of atom considered for calculating end-to-end distance

        bins: int
        The number of bins for bar plot 
        violin_plot: bool 
        If True box plot is visualized

        means: bool
        If True mean is showing in the box plot 

        median: bool
        If True median is showing in the box plot
        """
        visualization.end_to_end_distances_plot(self.trajectories, atom_selector, bins, violin_plot, means, median)

    def plot_asphericity_dist(self, bins = 50,violin_plot=True, means=False, median=True ):
        """
        Plot asphericity distribution in each ensemble.
        Asphericity is calculated based on the gyration tensor.  

        Parameters
        ----------

        bins: int
        The number of bins for bar plot 
        violint_plot: bool 
        If True box plot is visualized

        means: bool
        If True mean is showing in the box plot 

        median: bool
        If True median is showing in the box plot
        """
        visualization.plot_asphericity_dist(self.trajectories ,bins, violin_plot, means, median )

    def plot_prolateness_dist(self, bins=50, violin_plot=True, means=False, median=True):
        """
        Plot prolateness distribution in each ensemble.
        Prolateness is calculated based on the gyration tensor.  

        Parameters
        ----------

        bins: int
        The number of bins for bar plot 
        violint_plot: bool 
        If True box plot is visualized

        means: bool
        If True mean is showing in the box plot 

        median: bool
        If True median is showing in the box plot
        """
        visualization.plot_prolateness_dist(self.trajectories, bins, violin_plot, means, median)

    def plot_alpha_angles_dist(self, bins=50):

        """
        It plot the distribution of alpha angles.

        Parameters
        ----------
        bins : int
        The number of bins for bar plot 
        """
        visualization.plot_alpha_angles_dist(self.trajectories, bins)

    def plot_contact_prob(self,threshold = 0.8,dpi = 96):
        """
        It plots the contact probability map based on the threshold. 
        The default value for threshold is 0.8[nm], 

        Parameters
        ----------
        title: str 
        You need to specify a title for the plot

        threshold: float
        Determing the threshold fo calculating the contact frequencies. default value is 0.8[nm]

        dpi: int
        For changing the quality and dimension of the output figure
        """
        visualization.plot_contact_prob(self.trajectories,threshold,dpi)

    def plot_ramachandran_plot(self, two_d_hist=True, linespaces= (-180, 180, 80)):
        """
        It gets Ramachandran plot. If two_d_hist= True it returns 2D histogram 
        for each ensembles. If two_d_hist=False it returns a simple scatter plot 
        for ell ensembles in one plot.

        Parameters
        ----------

        two_d_hist: bool
        If True it returns 2D histogram for each ensemble. 

        linespaces: tuple
        You can customize the bins for 2D histogram
        """
        visualization.plot_ramachandran_plot(self.trajectories, two_d_hist, linespaces)
    
    def plot_ss_measure_disorder(self, pointer=None, figsize=(15,5)):
        """
        This function generates site specific flexibility parameter plot. For further information
        you can check this paper by G.Jeschke https://onlinelibrary.wiley.com/doi/epdf/10.1002/pro.4906. 
        In summary this score is sensitive to local flexibility based on the circular variance of the
        Ramachandran angles φ and ψ for each residue in the ensemble.

        This score ranges from 0 for identical dihederal angles for all conformers at the residue i to 1 
        for a uniform distribution of dihederal angles at the residue i. 

        Parameters
        ----------
        pointer: list 
        You can add the desired residues in a list and then you have a vertical dashed line to point those residues

        figsize:tuple
        You can change the size oof the figure here using a tuple. 
        """
        self._extract_features("phi_psi") # extract phi_psi features to calculate this score
        feature_dict = self.featurized_data # provide feature dictionary for plot function
        visualization.plot_ss_measure_disorder(feature_dict, pointer, figsize)

    def plot_ss_order_parameter(self, pointer=None, figsize=(15,5)):

        """
        This function generates site specific order parameter plot. For further information
        you can check this paper by G.Jeschke https://onlinelibrary.wiley.com/doi/epdf/10.1002/pro.4906. 
        In summary this score abstracts from local chain flexibility. The parameter is still site-specific, as orientation 
        correlations in IDRs and IDPs decrease with increasing sequence distance. 

        Parameters
        ----------
        pointer: list 
        You can add the desired residues in a list and then you have a vertical dashed line to point those residues

        figsize:tuple
        You can change the size oof the figure here using a tuple. 
        """

        visualization.plot_ss_order_parameter(self.trajectories, pointer, figsize)

    def plot_local_sasa(self, figsize=(15,5)):

        """
        This function plots the average amount of SASA for each residue through the ensemble

        Parameters
        ----------
        figsize: tuple \n
        You can change the size of plot here. 
        """
        visualization.plot_local_sasa(self, figsize)


    def plot_dist_ca_com(self, min_sep=2,max_sep=None, get_names=True, figsize=(6,2.5)):

        visualization.plot_dist_ca_com(self, min_sep, max_sep,get_names, figsize)


    ##################### PDF Reports #####################

    def generate_tsne_report(self):
        generate_tsne_report(self)
        
    def generate_dimenfix_report(self):
        generate_dimenfix_report(self)

    def generate_pca_report(self):
        generate_pca_report(self)

    def generate_custom_report(self):
        generate_custom_report(self)
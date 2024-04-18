from pathlib import Path
import re
import shutil
from typing import Dict
import zipfile
from dpet.api_client import APIClient
from dpet.ensemble import Ensemble
from dpet.visualization.reports import generate_custom_report, generate_dimenfix_report, generate_pca_report, generate_tsne_report
import dpet.visualization.visualization as visualization
from dpet.utils import extract_tar_gz
import os
import mdtraj
import numpy as np
from dpet.dimensionality_reduction.dimensionality_reduction import DimensionalityReductionFactory

class EnsembleAnalysis:
    def __init__(self, ens_codes:list[str], data_dir:str):
        self.data_dir = Path(data_dir)
        self.api_client = APIClient()
        self.feature_names = []
        self.all_labels = []
        self.ens_codes = ens_codes
        plot_dir = os.path.join(self.data_dir, visualization.PLOT_DIR)
        os.makedirs(plot_dir, exist_ok=True)
        self.figures = {}
        self.ensembles: Dict[str, Ensemble] = {}

    def __del__(self):
        if hasattr(self, 'api_client'):
            self.api_client.close_session()
    
    def download_from_ped(self):
        """
        Automate Downloading ensembles using PED API 

        Note
        ----
        The function only downloads ensembles in the PED ID format, which consists of a string starting with 'PED'
        followed by a numeric identifier and 'e' followed by another numeric identifier.
        Example: 'PED00423e001', 'PED00424e001'
        """
        ped_pattern = r'^(PED\d{5})(e\d{3})$'

        # Filter the ens_codes list using regex
        for ens_code in self.ens_codes:
            match = re.match(ped_pattern, ens_code)
            if not match:
                print(f"Entry {ens_code} does not match the PED ID pattern and will be skipped.")
                continue
            
            ped_id = match.group(1)
            ensemble_id = match.group(2)
            tar_gz_filename = f'{ens_code}.tar.gz'
            tar_gz_file = os.path.join(self.data_dir, tar_gz_filename)

            pdb_filename = f'{ens_code}.pdb'
            pdb_file = os.path.join(self.data_dir, pdb_filename)

            if not os.path.exists(tar_gz_file) and not os.path.exists(pdb_file):
                url = f'https://deposition.proteinensemble.org/api/v1/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb'
                print(f"Downloading entry {ens_code} from PED.")
                headers = {'accept': '*/*'}

                response = self.api_client.perform_get_request(url, headers=headers)
                if response:
                    # Download and save the response content to a file
                    self.api_client.download_response_content(response, tar_gz_file)
                    print(f"Downloaded file {tar_gz_filename} from PED.")
            else:
                print(f"Ensemble {ens_code} already downloaded. Skipping.")

            # Extract the .tar.gz file
            if not os.path.exists(pdb_file):
                extract_tar_gz(tar_gz_file, self.data_dir, pdb_filename)
                print(f"Extracted file {pdb_filename}.")
            else:
                print(f"File {pdb_filename} already exists. Skipping extraction.")
    
    def download_from_atlas(self):
        """ 
        Automate Downloading MD ensembles from Atlas. 

        Note:
        -----
        The function only downloads ensembles provided as PDB ID with a chain identifier separated by an underscore.
        Example: '3a1g_B'
        """
        pdb_pattern = r'^\d\w{3}_[A-Z]$'
        new_ens_codes_mapping = {}
        for ens_code in self.ens_codes:

            if not re.match(pdb_pattern, ens_code):
                print(f"Entry {ens_code} does not match the PDB ID pattern and will be skipped.")
                continue

            zip_filename = f'{ens_code}.zip'
            zip_file = os.path.join(self.data_dir, zip_filename)

            if not os.path.exists(zip_file):
                print(f"Downloading entry {ens_code} from Atlas.")
                url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{ens_code}/{ens_code}_protein.zip"
                headers = {'accept': '*/*'}

                response = self.api_client.perform_get_request(url, headers=headers)
                if not response:
                    continue
                # Download and save the response content to a file
                self.api_client.download_response_content(response, zip_file)
                print(f"Downloaded file {zip_filename} from Atlas.")
            else:
                print("File already exists. Skipping download.")

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Map reps to original ensemble code
                zip_contents = zip_ref.namelist()
                new_ens_codes = [fname.split('.')[0] for fname in zip_contents if fname.endswith('.xtc')]
                new_ens_codes_mapping[ens_code] = new_ens_codes
                # Unzip
                zip_ref.extractall(self.data_dir)
                print(f"Extracted file {zip_file}.")

            # Remove unused files.
            for unused_path in self.data_dir.glob("*.tpr"):
                os.remove(unused_path)
            os.remove(os.path.join(self.data_dir, "README.txt"))

            # Copy and rename topology file
            old_pdb_file = os.path.join(self.data_dir,f"{ens_code}.pdb")
            for new_code in new_ens_codes:
                new_pdb_file = os.path.join(self.data_dir,f"{new_code}.top.pdb")
                shutil.copy(old_pdb_file, new_pdb_file)
                print(f"Copied and renamed {old_pdb_file} to {new_pdb_file}.")
            # Delete old topology file
            os.remove(old_pdb_file)

        # Update self.ens_codes using the mapping
        updated_ens_codes = []
        for old_code in self.ens_codes:
            updated_ens_codes.extend(new_ens_codes_mapping.get(old_code, [old_code]))

        self.ens_codes = updated_ens_codes
        print("Analysing ensembles:", self.ens_codes)

    def download_from_database(self, database: str =None):
        """ 
        Download ensembles from databases.

        Parameters
        ----------
        database : str
            Choose the database you want to download from ('ped'/'atlas').

        Note
        ----
        For PED database:
            The function only downloads ensembles in the PED ID format, which consists of a string starting with 'PED'
            followed by a numeric identifier and 'e' followed by another numeric identifier.
            Example: 'PED00423e001', 'PED00424e001'

        For atlas database:
            The function only downloads ensembles provided as PDB ID with a chain identifier separated by an underscore.
            Example: '3a1g_B'
        """
        if database == "ped":
            self.download_from_ped()
        elif database == "atlas":
            self.download_from_atlas()

    def generate_trajectories(self):
        """
        Load trajectory files into mdtraj objects.
        
        Supported file formats:
            1. [ens_code].dcd (trajectory file) + [ens_code].top.pdb (topology file)
            2. [ens_code].xtc (trajectory file) + [ens_code].top.pdb (topology file)
            3. [ens_code].pdb
            4. Directory [ens_code] containing several .pdb files
        
        For each ensemble code (ens_code):
            - If both trajectory (.dcd or .xtc) and topology (.top.pdb) files exist, load the trajectory.
            - If only a .pdb file exists, generate trajectory and topology files from the .pdb file.
            - If a directory [ens_code] exists containing .pdb files, generate trajectory and topology files from the directory.

        Note
        ----
        Using 'download_from_database' transforms the downloaded data into the appropriate format.
        """
        for ens_code in self.ens_codes:
            ensemble = Ensemble(ens_code, self.data_dir)
            ensemble.load_trajectory()
            ensemble.select_chain()
            ensemble.check_coarse_grained()
            self.ensembles[ens_code] = ensemble
            
    def random_sample_trajectories(self, sample_size: int):
        """
        Sample a defined random number of conformations from the ensemble 
        trajectory. 

        Parameters
        ----------
        sample_size: int
            Number of conformations sampled from the ensemble. 
        """
        for ensemble in self.ensembles.values():
            ensemble.random_sample_trajectory(sample_size)

    def extract_features(self, featurization: str, normalize: bool = False, min_sep: int = 2, max_sep: int = None):
        """
        Extract the selected feature.

        Parameters
        ----------
        featurization : str
            Choose between "phi_psi", "ca_dist", "a_angle", "tr_omega", and "tr_phi".

        normalize : bool, optional
            Whether to normalize the data. Only applicable to the "ca_dist" method. Default is False.

        min_sep : int, optional
            Minimum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is 2.

        max_sep : int, optional
            Maximum separation distance for "ca_dist", "tr_omega", and "tr_phi" methods. Default is None.
        """
        self._featurize(featurization=featurization, min_sep=min_sep, max_sep=max_sep)
        self.concat_features = self._get_concat_features()
        self._create_all_labels()
        if normalize and featurization == "ca_dist":
            self._normalize_data()

    def _exists_coarse_grained(self):
        return any(ensemble.coarse_grained for ensemble in self.ensembles.values())

    def _featurize(self, featurization: str, min_sep, max_sep):
        if featurization in ("phi_psi", "tr_omega", "tr_phi") and self._exists_coarse_grained():
            raise ValueError(f"{featurization} feature extraction is not possible when working with coarse-grained models.")
        self.featurization = featurization
        for ensemble in self.ensembles.values():
            ensemble.extract_features(featurization, min_sep, max_sep)
        self.feature_names = list(self.ensembles.values())[0].names
        print("Feature names:", self.feature_names)

    def _create_all_labels(self):
        self.all_labels = []
        for ens_id, ensemble in self.ensembles.items():
            num_data_points = len(ensemble.features)
            self.all_labels.extend([ens_id] * num_data_points)

    def _normalize_data(self):
        mean = self.concat_features.mean(axis=0)
        std = self.concat_features.std(axis=0)
        self.concat_features = (self.concat_features - mean) / std
        for ensemble in self.ensembles.values():
            ensemble.normalize_features(mean, std)

    def _calculate_rg_for_trajectory(self, trajectory:mdtraj.Trajectory):
        return [mdtraj.compute_rg(frame) for frame in trajectory]

    @property
    def rg(self):
        """
        Calculates Rg for each conformations in the loaded ensembles.
        The returned values are in Angstrom.  
        """
        rg_values_list = []
        for ensemble in self.ensembles.values():
            traj = ensemble.trajectory
            rg_values_list.extend(self._calculate_rg_for_trajectory(traj))
        return [item[0] * 10 for item in rg_values_list]

    def _get_concat_features(self, fit_on: list[str]=None):
        if fit_on and any(f not in self.ens_codes for f in fit_on):
            raise ValueError("Cannot fit on ensembles that were not provided as input.")
        if fit_on is None:
            fit_on = self.ens_codes
        concat_features = [ensemble.features for ens_code, ensemble in self.ensembles.items() if ens_code in fit_on]
        concat_features = np.concatenate(concat_features, axis=0)
        print("Concatenated featurized ensemble shape:", concat_features.shape)
        return concat_features

    def reduce_features(self, method: str, fit_on:list[str]=None, *args, **kwargs):
        """
        Perform dimensionality reduction on the extracted features. The options are "pca", "tsne", "dimenfix", "mds" and "kpca".

        Parameters
        ----------
        method: str 
            Choose between pca", "tsne", "dimenfix", "mds" and "kpca".

        fit_on: list[str]
            if method is "pca" or "kpca" the fit_on parameter specifies on which ensembles the models should be fit. 
            The model will then be used to transform all ensembles.
        """
        self.reducer = DimensionalityReductionFactory.get_reducer(method, *args, **kwargs)
        self.reduce_dim_method = method
        if method in ("pca","kpca"):
            fit_on_data = self._get_concat_features(fit_on=fit_on)
            self.reduce_dim_model = self.reducer.fit(data=fit_on_data)
            self.reduce_dim_data = {}
            for ensemble in self.ensembles.values():
                ensemble.reduce_dim_data = self.reducer.transform(ensemble.features)
                print("Reduced dimensionality ensemble shape:", ensemble.reduce_dim_data.shape)
            self.transformed_data = self.reducer.transform(data=self.concat_features)
        else:
            self.transformed_data = self.reducer.fit_transform(data=self.concat_features)

    def execute_pipeline(self, featurization_params:dict, reduce_dim_params:dict, database:str=None, subsample_size:int=None):
        """
        Execute the data analysis pipeline end-to-end. The pipeline includes:
        1. Download from database (optional)
        2. Generate trajectories
        3. Sample a random number of conformations from trajectories (optional)
        4. Perform feature extraction
        5. Perform dimensionality reduction

        Parameters
        ----------
        featurization_params: dict
            Parameters for feature extraction. The only required parameter is "featurization",
            which can be "phi_psi", "ca_dist", "a_angle", "tr_omega" or "tr_phi". 
            Other method-specific parameters are optional.
        reduce_dim_params: dict
            Parameters for dimensionality reduction. The only required parameter is "method",
            which can be "pca", "tsne", "dimenfix", "mds" or "kpca".
        database: str, optional
            Optional parameter that specifies the database from which the entries should be downloaded.
            Options are "ped" and "atlas". Default is None.
        subsample_size: int, optional
            Optional parameter that specifies the trajectory subsample size. Default is None.
        """
        self.download_from_database(database)
        self.generate_trajectories()
        if subsample_size is not None:
            self.random_sample_trajectories(subsample_size)
        self.extract_features(**featurization_params)
        self.reduce_features(**reduce_dim_params)

    #----------------------------------------------------------------------
    #------------------- Integrated plot functions ------------------------
    #----------------------------------------------------------------------

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
        
        visualization.tsne_ramachandran_plot_density(self, save)

    def tsne_scatter_plot(self, save:bool=False):
        """
        Plot the results of t-SNE analysis. 
        Three scatter plot will be generated based on original, clustering and Rg labels. 
        One KDE density plot will also be generated to shod the most populated areas in 
        the reduced dimension.   

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        visualization.tsne_scatter_plot(self, save)

    def tsne_scatter_plot_rg(self, save:bool=False):
        # It plots a redundant plot which implemented in tsne_scatter_plot and could be removed
        visualization.tsne_scatter_plot_rg(self, save)

    def dimenfix_scatter(self, save:bool=False):
        """
        Plot the the complete results for dimenfix method. 

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
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

    def pca_cumulative_explained_variance(self, save:bool=False):

        """
        Plot the cumulative variance. Only applicable when the
        dimensionality reduction method is "pca"

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        visualization.pca_cumulative_explained_variance(self, save)
    
    def pca_plot_2d_landscapes(self, save:bool=False):

        """
        Plot 2D landscapes when the dimensionality reduction method 
        is "pca" or "kpca"

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        visualization.pca_plot_2d_landscapes(self, save)

    def pca_plot_1d_histograms(self, save:bool=False):
        """
        Plot 1D histogram when the dimensionality reduction method 
        is "pca" or "kpca"

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        visualization.pca_plot_1d_histograms(self, save)

    def pca_correlation_plot(self, num_residues:int,sel_dims:list[int]):
        visualization.pca_correlation_plot(num_residues, sel_dims, self)
    
    def pca_rg_correlation(self, save:bool=False):
        """
        Examine and plot the correlation between PC dimension 1 and the amount of Rg
        Typically high correlation can be detected here. 

        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
        """
        visualization.pca_rg_correlation(self, save)
        
    def plot_global_sasa(self, save=False, showmeans=True ,showmedians=True):

        """
        Plot the distribution of SASA for each conformation 
        within the ensembles.

        Parameters
        ----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.

        showmean: bool
            if True it will show the mean 

        showmedian: bool
            if True it will show the median
        """
        visualization.plot_global_sasa(self, save, showmeans, showmedians)

    def plot_rg_vs_asphericity(self, save=False):
        """
        Plot the Rg versus Asphericity and gives the pearson correlation coefficient to evaluate 
        the correlation between Rg and Asphericity. 
        """
        visualization.plot_rg_vs_asphericity(self, save)
    '''
    def trajectories_plot_density(self):
        visualization.trajectories_plot_density(self.trajectories)
    '''

    def plot_rg_vs_prolateness(self, save=False):
        """
        Plot the Rg versus Prolateness and gives the pearson correlation coefficient to evaluate 
        the correlation between Rg and Prolateness. 
        
        Parameters
        -----------
        save: bool, optional
            If True the plot will be saved in the data directory. Default is False.
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
        if self._exists_coarse_grained():
            print("This analysis is not possible with coarse-grained models.")
            return
        visualization.plot_relative_helix_content(self.ensembles)

    def trajectories_plot_rg_comparison(self, n_bins:int=50, bins_range:tuple=(1, 4.5), dpi:int=96):
        """
        Plot the distribution of the Rg whithin each ensemble
        
        Parameter 
        ---------
        n_bins : int 
        bins_range : tuple
        change the Rg scale in x-axis 
        dpi : int
        """
        visualization.trajectories_plot_rg_comparison(self.ensembles, n_bins, bins_range, dpi)

    def plot_average_dmap_comparison(self, 
                                    ticks_fontsize:int=14,
                                    cbar_fontsize:int=14,
                                    title_fontsize:int=14,
                                    dpi:int=96,
                                    max_d:float=6.8,
                                    use_ylabel:bool=True):
        
        """
        Plot the average distance maps for selected ensembles.
        
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
        
        visualization.plot_average_dmap_comparison(self.ensembles, ticks_fontsize, cbar_fontsize, title_fontsize, dpi, max_d, use_ylabel)

    def plot_cmap_comparison(self,
                            title:str,
                            ticks_fontsize:int=14,
                            cbar_fontsize:int=14,
                            title_fontsize:int=14,
                            dpi:int=96,
                            cmap_min:float=-3.5,
                            use_ylabel:bool=True):
        visualization.plot_cmap_comparison(self.trajectories, title, ticks_fontsize, cbar_fontsize, title_fontsize, dpi, cmap_min, use_ylabel)

    def plot_distance_distribution_multiple(self, dpi:int=96):
        visualization.plot_distance_distribution_multiple(self.trajectories, dpi)

    def end_to_end_distances_plot(self, bins:int=50, violin_plot:bool=True, means:bool=False, median:bool=True):
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
        visualization.end_to_end_distances_plot(self.ensembles, bins, violin_plot, means, median)

    def plot_asphericity_dist(self, bins:int=50,violin_plot:bool=True, means:bool=False, median:bool=True):
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
        visualization.plot_asphericity_dist(self.ensembles ,bins, violin_plot, means, median )

    def plot_prolateness_dist(self, bins:int=50, violin_plot:bool=True, means:bool=False, median:bool=True):
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
        visualization.plot_prolateness_dist(self.ensembles, bins, violin_plot, means, median)

    def plot_alpha_angles_dist(self, bins:int=50):

        """
        Plot the distribution of alpha angles.

        Parameters
        ----------
        bins : int
            The number of bins for bar plot 
        """
        visualization.plot_alpha_angles_dist(self.ensembles, bins)

    def plot_contact_prob(self,title:str, threshold:float=0.8, dpi:int=96):
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
        """
        if self._exists_coarse_grained():
            print("This analysis is not possible with coarse-grained models.")
            return
        visualization.plot_contact_prob(self.ensembles,title,threshold,dpi)

    def plot_ramachandran_plot(self, two_d_hist:bool=True, linespaces:tuple=(-180, 180, 80)):
        """
        Ramachandran plot. If two_d_hist= True it returns 2D histogram 
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
    
    def plot_ss_measure_disorder(self, pointer:list=None, figsize:tuple=(15,5)):
        """
        Generate site specific flexibility parameter plot. Further information is available in
        this paper by G.Jeschke https://onlinelibrary.wiley.com/doi/epdf/10.1002/pro.4906. 
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
        self.perform_feature_extraction("phi_psi") # extract phi_psi features to calculate this score
        #feature_dict = self.featurized_data # provide feature dictionary for plot function
        visualization.plot_ss_measure_disorder(self.ensembles, pointer, figsize)

    def plot_ss_order_parameter(self, pointer:list=None, figsize:tuple=(15,5)):

        """
        Generate site specific order parameter plot. For further information
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

        visualization.plot_ss_order_parameter(self.ensembles, pointer, figsize)

    #----------------------------------------------------------------------
    #------------- Functions for generating PDF reports -------------------
    #----------------------------------------------------------------------

    def generate_custom_report(self):
        
        """
        Generate pdf report with all plots that were explicitly called during the session.
        """

        generate_custom_report(self)

    def generate_report(self):

        """
        Generate pdf report with all plots relevant to the conducted analysis.
        """

        if self.reduce_dim_method == "tsne":
            generate_tsne_report(self)
        if self.reduce_dim_method == "dimenfix":
            generate_dimenfix_report(self)
        if self.reduce_dim_method == "pca":
            generate_pca_report(self)
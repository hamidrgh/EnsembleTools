import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dpet.visualization.visualization import PLOT_DIR, dimenfix_cluster_scatter_plot, dimenfix_cluster_scatter_plot_2, dimenfix_scatter_plot_rg, dimenfix_scatter_plot_ens, pca_cumulative_explained_variance, pca_plot_1d_histograms, pca_plot_2d_landscapes, pca_rg_correlation, tsne_ramachandran_plot_density, tsne_scatter_plot, tsne_scatter_plot_rg

def generate_tsne_report(analysis):
    pdf_file_path = os.path.join(analysis.data_dir, PLOT_DIR, 'tsne.pdf')
    with PdfPages(pdf_file_path) as pdf:
        fig = tsne_ramachandran_plot_density(analysis, False)
        pdf.savefig(fig)
        plt.close(fig)

        fig = tsne_scatter_plot(analysis, False)
        pdf.savefig(fig)
        plt.close(fig)

        fig = tsne_scatter_plot_rg(analysis, False)

        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Plots saved to {pdf_file_path}")

def generate_dimenfix_report(analysis):
    pdf_file_path = os.path.join(analysis.data_dir, PLOT_DIR, 'dimenfix.pdf')
    with PdfPages(pdf_file_path) as pdf:
        fig = dimenfix_scatter_plot_rg(analysis)
        pdf.savefig(fig)
        plt.close(fig)

        fig = dimenfix_scatter_plot_ens(analysis)
        pdf.savefig(fig)
        plt.close(fig)

        fig = dimenfix_cluster_scatter_plot(analysis)
        pdf.savefig(fig)
        plt.close(fig)

        fig = dimenfix_cluster_scatter_plot_2(analysis)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Plots saved to {pdf_file_path}")

def generate_pca_report(analysis):
    pdf_file_path = os.path.join(analysis.data_dir, PLOT_DIR, 'pca.pdf')
    with PdfPages(pdf_file_path) as pdf:
    
        fig = pca_cumulative_explained_variance(analysis)
        pdf.savefig(fig)
        plt.close(fig)

        fig = pca_plot_2d_landscapes(analysis)
        pdf.savefig(fig)
        plt.close(fig)

        fig = pca_plot_1d_histograms(analysis)
        pdf.savefig(fig)
        plt.close(fig)

        fig = pca_rg_correlation(analysis)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Plots saved to {pdf_file_path}")

def generate_custom_report(analysis):
    pdf_file_path = os.path.join(analysis.data_dir, PLOT_DIR, 'custom_report.pdf')
    with PdfPages(pdf_file_path) as pdf:
        for fig in analysis.figures.values():
            pdf.savefig(fig)
    print(f"Plots saved to {pdf_file_path}")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dpet.visualization.visualization import tsne_ramachandran_plot_density, tsne_scatter_plot, tsne_scatter_plot_rg

def generate_tsne_report(plot_dir, concat_features, bestK, bestP, best_kmeans, besttsne, all_labels, ens_codes, rg):
    pdf_file_path = plot_dir + '/tsne.pdf'
    with PdfPages(pdf_file_path) as pdf:
        fig = tsne_ramachandran_plot_density(plot_dir, concat_features, bestP, bestK, best_kmeans, False)

        pdf.savefig(fig)
        plt.close(fig)

        fig = tsne_scatter_plot(plot_dir, all_labels, ens_codes, rg, bestK, bestP, best_kmeans, besttsne, False)
        pdf.savefig(fig)
        plt.close(fig)

        fig = tsne_scatter_plot_rg(rg, besttsne, plot_dir, bestP, bestK, False)

        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Plots saved to {pdf_file_path}")
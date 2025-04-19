import os

import matplotlib.pyplot as plt

class Plotter():
    @staticmethod
    def save_corr_plot(data, title, path):
        plt.hist(data, bins=20, edgecolor='k', alpha=0.7)
        plt.xlabel('Explained variance')
        plt.ylabel('# neurons')
        plt.title(title)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path)
        plt.close()
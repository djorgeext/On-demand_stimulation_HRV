import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import zscore

def psd(serie, window_size=2048):
        overlap = window_size // 2
        quantity = len(serie) // overlap
        cutting = quantity * overlap
        serie_right = np.reshape(serie[:cutting], (quantity, overlap))
        serie_right = np.concatenate((serie_right[:-1], serie_right[1:]), axis=1)
        serie_left = np.reshape(np.flip(serie)[:cutting], (quantity, overlap))
        serie_left = np.concatenate((serie_left[:-1], serie_left[1:]), axis=1)
        serie_matrix = np.concatenate((serie_right, serie_left), axis=0)
        serie_matrix = zscore(serie_matrix, axis=1)
        serie_matrix = np.abs(np.fft.fft(serie_matrix, axis=1))**2
        return np.mean(serie_matrix, axis=0)[:window_size//2 + 1]

path_list = [
    "16539",
    "16786",
    "008",
    "16420",
    "16483",
    "003",
    "007",
    "nsr041RRcl",
    "nsr013RRcl",
    "nsr033RRcl",
    "nsr018RRcl",
    "nsr034RRcl",
    "nsr003RRcl",
    "nsr026RRcl",
    "18177",
    "16273",
    "nsr054RRcl",
    "006",
    "nsr048RRcl",
    "19088",
    "000",
    "nsr010RRcl",
    "nsr038RRcl",
    "nsr022RRcl",
    "nsr045RRcl",
    "nsr016RRcl"
]
series_path = 'series/'
seeds = [7, 101, 211, 317, 421]
percents_array = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
percents_dirs = [f"percent_{int(percent * 100)}" for percent in percents_array]
freqs = np.linspace(0, 0.5, 2049)

for subject in path_list:
    subject_path = os.path.join(series_path, subject+".txt")
    original_serie = np.loadtxt(subject_path, dtype=float)
    
    for seed in seeds:
        for percent_dir in percents_dirs:
            imputed_file = os.path.join(subject, str(seed), percent_dir, f"modified_serie_filled.txt")
            if os.path.exists(imputed_file):
                imputed_serie = np.loadtxt(imputed_file, dtype=float)
                
                # It is safer to assign the figure to a variable to close it explicitly
                fig = plt.figure(figsize=(14, 7))
                plt.plot(original_serie, label='Original', color='blue')
                plt.plot(imputed_serie, label='Imputed', color='green')
                plt.title(f'Subject: {subject}, Percent: {percent_dir.split("_")[1]}%, Seed: {seed}')
                plt.xlabel('Sample Index')
                plt.ylabel('RR Interval Value')
                plt.legend()
                plt.grid()
                plt.tight_layout()

                output_dir = os.path.join(subject, str(seed), percent_dir)
                output_file = os.path.join(output_dir, f'series_comparison_seed_{seed}.png')
                plt.savefig(output_file)
                
                # Close the specific figure to free memory completely
                plt.close(fig)

    print(f"Finished processing subject: {subject}")
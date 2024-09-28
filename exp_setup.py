######################################################## 
######################## IMPORTS #######################
########################################################
from knn_chain import knn_chain

from scipy.cluster.hierarchy import ward
from scipy.spatial.distance import pdist

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import mmread

from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

# Time complexity analysis packages
import pstats, cProfile
import pyximport
pyximport.install()

# Memory complexity analysis packages
import tracemalloc

# Datasets
import tensorflow as tf


######################################################## 
######################## INPUTS ########################
########################################################

n = 5
ks = [1, 3, 10, 15, 25, 35]
all_ks = [1, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 90]
linear_ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Load synthetic datasets
s_X = np.loadtxt("datasets/generated_smallX")
med_X = np.loadtxt("datasets/generated_mediumX")
large_X = np.loadtxt("datasets/generated_largeX")
xl_X = np.loadtxt("datasets/generated_xlargeX")
xxl_X = np.loadtxt("datasets/generated_xxlargeX")

dataset_sizes_syn = [996, 5000, 9990, 15000, 19980]

# Load mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
n_images = x_train.shape[0]
X_images = x_train.reshape(n_images, 28, 28)
mnist_data = X_images.reshape(n_images, 28 * 28)[:30000]

# Load BBC news
matrix = mmread('datasets/bbc.mtx')
bbc_matrix = matrix.toarray().transpose()

# tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
# bbc_tsne = tsne.fit_transform(bbc_matrix)
colours = ListedColormap(['#FFB000', '#FE6100', '#DC267F', '#785EF0', '#648FFF'])
classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# Load Adult 
adult = np.loadtxt("datasets/adult_train.data")
adult = np.nan_to_num(adult)

dataset_sizes_real = [30000, 2225, 34190]

# adult_tsne = tsne.fit_transform(adult)

# Titles
syn_titles = ['Synthetic Dataset 1','Synthetic Dataset 2','Synthetic Dataset 3','Synthetic Dataset 4','Synthetic Dataset 5']
real_titles = ['MNIST','BBC News','Adult']


################### OPTIMAL K INPUTS ###################
init = [0, 1]
n = 7
for i in range(2,n+1):
    last_elem = init[-1]
    new_elem = last_elem+i
    mirror=[x + new_elem for x in init]
    
    init = init + mirror
    
optimal_ds = [[x, 0] for x in init]

timings = [0.000336241838, 0.00034539718200000003, 0.0003465008039999998, 0.00036066142199999985, 0.000368905304, 0.00037725686599999987, 0.0003843531960000001, 0.0003915828280000003, 0.00039984054000000006,  0.00041048484600000033]
opt_timings = [x*1000 for x in timings] # convert to ms 


#######################################################
###################### FUNCTIONS ######################
#######################################################

# Recording complexity
def record_runtimes(X, ks, n = 20):
    avgs = []
    for k in ks:
        avg = 0.0
        for i in range(n):
            cProfile.runctx("knn_chain(X, k)", globals(), locals(), "Profile.prof")
            k1 = pstats.Stats("Profile.prof")
            avg += k1.strip_dirs().sort_stats("time").total_tt

        avg = avg/n
        avgs.append(avg)
    return avgs

def record_memory(X, ks, n):
    mems = []
    for k in ks:
        mem = 0.0
        for _ in range(n):
            tracemalloc.start()
            knn_chain(X, k)
            mem += tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        mems.append(mem/n)
    return mems

def record_nn_runtimes(datasets, n):
    time = []
    for ds in datasets:
        avg = 0
        for _ in range(n):

            cProfile.runctx("pdist(ds)", globals(), locals(), "Profile.prof")
            k1 = pstats.Stats("Profile.prof")
            avg += k1.strip_dirs().sort_stats("time").total_tt

            y = pdist(ds)

            cProfile.runctx("ward(y)", globals(), locals(), "Profile.prof")
            k1 = pstats.Stats("Profile.prof")
            avg += k1.strip_dirs().sort_stats("time").total_tt
            
        avg = avg/n
        time.append(avg)
    return time

def record_nn_mem(datasets, n):
    mems = []
    for ds in datasets:
        mem = 0
        for _ in range(n):
            tracemalloc.start()
            y = pdist(ds) 
            ward(y)
            mem += tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        mem = mem/n
        mems.append(mem)
    return mems

# Plotting
def plot_runtimes(ks, avgs):
    str_ks = list(map(str, ks))
    plt.figure(figsize=(8, 6))
    plt.plot(str_ks, avgs, marker='o', c = '#D81B60', alpha=0.7)
    
    for i, avg in enumerate(avgs):
        plt.text(str_ks[i], avg, f'{avg:.4f}', ha='center', va='bottom', fontsize=10, color='#000000',rotation=40, fontweight='bold')
    
    plt.title("Execution time comparison")
    plt.xlabel("Value of k")
    plt.xticks(ticks=str_ks, labels=str_ks) 
    plt.ylabel("Execution time in seconds")
    plt.grid(c = "#d3d3d3")
    plt.ylim(0, max(avgs) * 1.3)
    plt.show()


def plot_all_runtimes(times, ks, titles, label_indices):
    n = len(titles)  
    colours = ['#FFB000', '#FE6100', '#DC267F', '#785EF0', '#648FFF']
    
    cols = 3
    rows = (n + cols - 1) // cols 
    fig, axes = plt.subplots(rows, cols, figsize=(45, 15), sharex=False, sharey=False)
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        indices = list(range(len(ks))) 
        axes[i].plot(indices, times[i], marker='o', color=colours[i % len(colours)])

        for j, avg in enumerate(times[i]):
            if j == label_indices[i]:
                axes[i].text(indices[j], avg, f'{avg:.3f}', ha='center', va='top', fontsize=25, color='#000000', rotation = 50, fontweight='bold')
            else:
                axes[i].text(indices[j], avg, f'{avg:.3f}', ha='center', va='bottom', fontsize=25, color='#000000', rotation = 50, fontweight='bold')

        axes[i].set_title(titles[i], fontsize=40)
        axes[i].set_xlabel('k', fontsize=30)
        axes[i].set_ylabel('Average Runtime (s)', fontsize=30)
        axes[i].grid(c="#d3d3d3")
        axes[i].set_ylim(bottom=0, top = max(times[i])*1.2)
        axes[i].tick_params(axis='y', labelsize=20)
        axes[i].set_xticks(indices)
        axes[i].set_xticklabels([str(k) for k in ks], fontsize=20)

    for i in range(n, rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_comparison(nn_chain_values, knn_chain_values, opt_values, x_labels, size=(12, 8)):
    n = len(nn_chain_values)
    fig, axes = plt.subplots(1, n, figsize=size, sharey=False)
    barWidth = 0.35 

    for i in range(n):
        br1 = np.arange(1)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]

        axes[i].bar(br1, nn_chain_values[i], color='#FFB000', width=.2, label='NN Chain', alpha=0.8)
        axes[i].bar(br2, knn_chain_values[i], color='#785EF0', width=.2, label='K-NN Chain', alpha=0.8)
        axes[i].bar(br3, opt_values[i], color='#DC267F', width=.2, label='Optimal-NN Chain', alpha=0.8)

        axes[i].set_title(f'{x_labels[i]}', fontsize=20, pad=10)
        axes[i].set_ylabel('Execution Time', fontweight='bold', fontsize=12, labelpad=10)

        axes[i].text(br1[0], nn_chain_values[i], f'{nn_chain_values[i]:.4f}', ha='center', fontsize=11)
        axes[i].text(br2[0], knn_chain_values[i], f'{knn_chain_values[i]:.4f}', ha='center', fontsize=11)
        axes[i].text(br3[0], opt_values[i], f'{opt_values[i]:.4f}', ha='center', fontsize=11)

        axes[i].set_xticks([br1[0] + barWidth / 40, br2[0] + barWidth / 40, br3[0] + barWidth / 40])
        axes[i].set_xticklabels(['NN Chain', '1-NN Chain', 'Optimal-NN Chain'], fontsize=10)
        axes[i].tick_params(axis='x', labelsize=10)

        axes[i].grid(True, which='both', color="#d3d3d3", linewidth=0.5, alpha=0.7)

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def human_format(x, pos):
    if x >= 1e9:
        return f'{x*1e-9:.1f}B'
    elif x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.1f}K'
    else:
        return f'{x:.0f}'

def plot_mem(nn_chain_results, knn_chain_results, dataset_sizes, x_title, x_labels, size = (12, 8)):
    barWidth = 0.25
    fig, ax = plt.subplots(figsize=size)

    br1 = np.arange(len(nn_chain_results))
    br2 = [x + barWidth for x in br1]

    ax.bar(br1, nn_chain_results, color='#FFB000', width=barWidth, label='NN Chain', alpha=0.8)
    ax.bar(br2, knn_chain_results, color='#785EF0', width=barWidth, label='KNN Chain', alpha=0.8)
    ax.scatter((br1+br2)/2, dataset_sizes, color='#DC267F', marker="^", s = 100, label='Dataset Size')

    ax.set_title('Memory usage of NN Chain vs KNN Chain', fontsize=18, pad=20)
    ax.set_xlabel(x_title, fontweight='bold', fontsize=15, labelpad=15)
    ax.set_ylabel('Memory Usage', fontweight='bold', fontsize=15, labelpad=15)
    ax.set_xticks([r + barWidth / 2 for r in range(len(nn_chain_results))])
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))

    # Add value labels on top of the bars (in human-readable format)
    for i in range(len(nn_chain_results)):
        ax.text(br1[i], nn_chain_results[i] * 1.1, human_format(nn_chain_results[i], None), ha='center', fontsize=9)
        ax.text(br2[i], knn_chain_results[i] * 1.1, human_format(knn_chain_results[i], None), ha='center', fontsize=9)
        ax.text((br1[i] + br2[i]) / 2, dataset_sizes[i] * 1.1, human_format(dataset_sizes[i], None), ha='center', fontsize=9, color='#DC267F')#'#785EF0', '#648FFF'

    ax.grid(True, which='both', axis='y', c = "#d3d3d3", linewidth=0.5, alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()


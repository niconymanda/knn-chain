######################################################### 
######################## IMPORTS ########################
#########################################################
import numpy as np
import json

# Inputs
from exp_setup import n, ks, all_ks, linear_ks, s_X, med_X, large_X, xl_X, xxl_X, mnist_data, bbc_matrix, adult

# Functions
from exp_setup import record_runtimes, record_nn_runtimes, record_memory, record_nn_mem

######################################################### 
############## TIME COMPLEXITY EXPERIMENTS ##############
######################################################### 
# 1. Synthetic Data
avgs_small_gen = record_runtimes(s_X, all_ks, n)
avgs_med_gen = record_runtimes(med_X, all_ks, n)
avgs_large_gen = record_runtimes(large_X, all_ks, n)
avgs_xl_gen = record_runtimes(xl_X, all_ks, n)
avgs_xxl_gen = record_runtimes(xxl_X, all_ks, n)

# 2. Real Data
avgs_mnist = record_runtimes(mnist_data, all_ks, n)
avgs_bbc = record_runtimes(bbc_matrix, all_ks, n)
avgs_adult = record_runtimes(adult, all_ks, n)

# 3. Compare to NN Chain (SciPy)
syn_nn_time = record_nn_runtimes([s_X, med_X, large_X, xl_X, xxl_X], n)
real_nn_time = record_nn_runtimes([mnist_data, bbc_matrix, adult], n)

######################################################### 
############# MEMORY COMPLEXITY EXPERIMENTS #############
######################################################### 
# 1. Synthetic Data
s_mem = record_memory(s_X, all_ks, n)
med_mem = record_memory(med_X, all_ks, n)
large_mem = record_memory(large_X, all_ks, n)
xl_mem = record_memory(xl_X, all_ks, n)
xxl_mem = record_memory(xxl_X, all_ks, n)

# 2. Real Data
mnist_mem = record_memory(mnist_data, all_ks, n)
bbc_mem = record_memory(bbc_matrix, all_ks, n)
adult_mem = record_memory(adult, all_ks, n)

# 3. Compare to NN Chain (SciPy)
syn_nn_mems = record_nn_mem([s_X, med_X, large_X, xl_X, xxl_X], n)
real_nn_mems = record_nn_mem([mnist_data, bbc_matrix, adult], n)

######################################################### 
################ SAVE EXPERIMENT OUTPUTS ################
######################################################### 
output = {}

output['time'] = {
                'avgs_small_gen' : avgs_small_gen,
                'avgs_med_gen' : avgs_med_gen,
                'avgs_large_gen' : avgs_large_gen,
                'avgs_xl_gen' : avgs_xl_gen,
                'avgs_xxl_gen' : avgs_xxl_gen,
                
                'avgs_mnist' : avgs_mnist,
                'avgs_bbc' : avgs_bbc,
                'avgs_adult' : avgs_adult,
                
                'syn_nn_time' : syn_nn_time,
                'real_nn_time' : real_nn_time
                }

output['memory'] = {
                    's_mem' : s_mem,
                    'med_mem' : med_mem,
                    'large_mem' : large_mem,
                    'xl_mem' : xl_mem,
                    'xxl_mem' : xxl_mem,
                    
                    'mnist_mem' : mnist_mem,
                    'bbc_mem' : bbc_mem,
                    'adult_mem' : adult_mem,
                    
                    'syn_nn_mems' : syn_nn_mems,
                    'real_nn_mems' : real_nn_mems
                    }

with open('exp_output.json', 'w') as f:
    json.dump(output, f)
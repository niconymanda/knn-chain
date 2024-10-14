######################################################### 
######################## IMPORTS ########################
#########################################################
import pandas as pd
import numpy as np
import json
# Inputs
from exp_setup import n, ks, all_ks, linear_ks, s_X, med_X, large_X, xl_X, xxl_X, mnist_data, bbc_matrix, adult, cadata, blood_transfusion, fmnist, white_wine, red_wine
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

# # 2. Real Data
avgs_blood = record_runtimes(blood_transfusion, all_ks, n)
avgs_bbc = record_runtimes(bbc_matrix, all_ks, n)
avgs_white_wine = record_runtimes(white_wine, all_ks, n)
avgs_red_wine = record_runtimes(red_wine, all_ks, n)
avgs_ca = record_runtimes(cadata, all_ks, n)
avgs_mnist = record_runtimes(mnist_data, all_ks, n)
avgs_fmnist = record_runtimes(fmnist, all_ks, n)
avgs_adult = record_runtimes(adult, all_ks, n)
avgs_bike = record_runtimes(bike, all_ks, n)

# 3. Compare to NN Chain (SciPy)
syn_nn_time = record_nn_runtimes([s_X, med_X, large_X, xl_X, xxl_X], n)
real_nn_time = record_nn_runtimes([blood_transfusion, bbc_matrix, white_wine, red_wine, cadata, fmnist, mnist_data, adult, bike], n)

######################################################### 
############# MEMORY COMPLEXITY EXPERIMENTS #############
######################################################### 
# 1. Synthetic Data
s_mem = record_memory(s_X, ks, n)
med_mem = record_memory(med_X, ks, n)
large_mem = record_memory(large_X, ks, n)
xl_mem = record_memory(xl_X, ks, n)
xxl_mem = record_memory(xxl_X, ks, n)

# 2. Real Data
blood_mem = record_memory(blood_transfusion, ks, n)
bbc_mem = record_memory(bbc_matrix, ks, n)
white_wine_mem = record_memory(white_wine, ks, n)
red_wine_mem = record_memory(red_wine, ks, n)
mnist_mem = record_memory(mnist_data, ks, n)
fmnist_mem = record_memory(fmnist, ks, n)
adult_mem = record_memory(adult, ks, n)
ca_mem = record_memory(cadata, ks, n)
bike_mem = record_memory(bike, ks, n)

# 3. Compare to NN Chain (SciPy)
syn_nn_mems = record_nn_mem([s_X, med_X, large_X, xl_X, xxl_X], n)
real_nn_mems = record_nn_mem([blood_transfusion, bbc_matrix, white_wine, red_wine, cadata, fmnist, mnist_data, adult, bike], n)

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
                
                'avgs_blood' : avgs_blood,
                'avgs_bbc' : avgs_bbc,
                'avgs_white_wine': avgs_white_wine,
                'avgs_red_wine': avgs_red_wine,
                'avgs_ca' : avgs_ca,
                'avgs_mnist' : avgs_mnist,
                'avgs_fmnist' : avgs_fmnist,
                'avgs_adult' : avgs_adult,
                'avgs_bike' : avgs_bike,
                
                'syn_nn_time' : syn_nn_time,
                'real_nn_time' : real_nn_time
                }


output['memory'] = {
                    's_mem' : s_mem,
                    'med_mem' : med_mem,
                    'large_mem' : large_mem,
                    'xl_mem' : xl_mem,
                    'xxl_mem' : xxl_mem,
                    
                    'blood_mem' : blood_mem,
                    'bbc_mem' : bbc_mem,
                    'white_wine_mem' : white_wine_mem,
                    'red_wine_mem' : red_wine_mem,
                    'ca_mem' : ca_mem,
                    'mnist_mem' : mnist_mem,
                    'fmnist_mem' : fmnist_mem,
                    'adult_mem' : adult_mem,
                    'bike_mem' : bike_mem,
                    
                    'syn_nn_mems' : syn_nn_mems,
                    'real_nn_mems' : real_nn_mems
                    }

with open('exp_output.json', 'w') as f:
    json.dump(output, f)
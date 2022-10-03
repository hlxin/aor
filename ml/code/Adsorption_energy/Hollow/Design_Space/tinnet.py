from __future__ import print_function, division

import numpy as np

from ase.db import connect

from tinnet.regression.regression import Regression

lr = 0.0007277509272133079
lamb = 0.1
atom_fea_len = 177
n_conv = 5
h_fea_len = 33
n_h = 5
Esp = -4.24

db = connect('design_space.db')

images = np.array([r.toatoms()
                   for r in db.select(purpose='adsorption_energy',
                                      site='hollow')])

ae = np.array([r['data']['ae']
               for r in db.select(purpose='adsorption_energy',
                                  site='hollow')])

vad2 = np.array([r['data']['vad2']
                 for r in db.select(purpose='adsorption_energy',
                                    site='hollow')])

check_ans_train_mae = np.zeros((10,10))
check_ans_train_mse = np.zeros((10,10))
check_ans_val_mae = np.zeros((10,10))
check_ans_val_mse = np.zeros((10,10))
check_ans_test_mae = np.zeros((10,10))
check_ans_test_mse = np.zeros((10,10))

for idx_test in range(0,10):
    for idx_validation in range(0,10):
        
        model = Regression(images,
                           ae,
                           task='test',
                           data_format='test',
                           phys_model='newns_anderson_semi', # for training
                           batch_size=1024,
                           idx_validation=idx_validation,
                           idx_test=idx_test,
                           atom_fea_len=atom_fea_len, # for architecture
                           n_conv=n_conv, # for architecture
                           h_fea_len=h_fea_len, # for architecture
                           n_h=n_h, # for architecture
                           Esp=Esp, # for TinNet
                           vad2=vad2, # for TinNet
                           emax=15, # for TinNet
                           emin=-15, # for TinNet
                           num_datapoints=3001 # for TinNet
                           )
        
        check_ans_train_mae[idx_test,idx_validation],\
        check_ans_train_mse[idx_test,idx_validation],\
        check_ans_val_mae[idx_test,idx_validation],\
        check_ans_val_mse[idx_test,idx_validation],\
        check_ans_test_mae[idx_test,idx_validation],\
        check_ans_test_mse[idx_test,idx_validation] = model.check_loss()

np.savetxt('check_ans_train_mae.txt', check_ans_train_mae)
np.savetxt('check_ans_train_mse.txt', check_ans_train_mse)
np.savetxt('check_ans_val_mae.txt', check_ans_val_mae)
np.savetxt('check_ans_val_mse.txt', check_ans_val_mse)
np.savetxt('check_ans_test_mae.txt', check_ans_test_mae)
np.savetxt('check_ans_test_mse.txt', check_ans_test_mse)

from __future__ import print_function, division

import numpy as np
import multiprocessing

from ase.db import connect

from piml.feature.voronoi import Voronoi
from piml.regression.regression import Regression

lr = 0.01
atom_fea_len = 64
n_conv = 3
h_fea_len = 128
n_h = 1

descriptor = Voronoi(max_num_nbr=12,
                     radius=8,
                     dmin=0,
                     step=0.2,
                     dict_atom_fea=None)

db = connect('design_space.db')

images = np.array([r.toatoms()
                   for r in db.select(purpose='classification',
                                      site='hollow')])

reconstructed = np.array([r['reconstructed']
                          for r in db.select(purpose='classification',
                                             site='hollow')])

features = multiprocessing.Pool().map(descriptor.feas, images)

check_ans_train_mae = np.zeros((10,10))
check_ans_train_mse = np.zeros((10,10))
check_ans_val_mae = np.zeros((10,10))
check_ans_val_mse = np.zeros((10,10))
check_ans_test_mae = np.zeros((10,10))
check_ans_test_mse = np.zeros((10,10))

for idx_test in range(0,10):
    for idx_validation in range(0,10):
        
        model = Regression(features,
                           reconstructed,
                           phys_model='gnn',
                           optim_algorithm='AdamW',
                           weight_decay=0.0001,
                           idx_validation=idx_validation,
                           idx_test=idx_test,
                           lr=lr,
                           atom_fea_len=atom_fea_len,
                           n_conv=n_conv,
                           h_fea_len=h_fea_len,
                           n_h=n_h,
                           batch_size=1024)
        
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

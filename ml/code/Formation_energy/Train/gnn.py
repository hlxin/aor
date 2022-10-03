from __future__ import print_function, division

import numpy as np
import multiprocessing

from ase.db import connect

from piml.feature.voronoi import Voronoi
from piml.regression.regression import Regression

lr = 0.009405740333386713
atom_fea_len = 237
n_conv = 6
h_fea_len = 67
n_h = 2

descriptor = Voronoi(max_num_nbr=12,
                     radius=8,
                     dmin=0,
                     step=0.2,
                     dict_atom_fea=None)

db = connect('train.db')

images = np.array([r.toatoms()
                   for r in db.select(purpose='formation_energy')])

fe = np.array([r['fe'] for r in db.select(purpose='formation_energy')])

features = multiprocessing.Pool().map(descriptor.feas, images)

final_ans_val_mae = np.zeros(10)
final_ans_val_mse = np.zeros(10)
final_ans_test_mae = np.zeros(10)
final_ans_test_mse = np.zeros(10)

for idx_test in range(0,10):
    for idx_validation in range(0,10):
        
        model = Regression(features,
                           fe,
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
                           batch_size=64)
        
        final_ans_val_mae[idx_test], final_ans_val_mse[idx_test],\
            final_ans_test_mae[idx_test], final_ans_test_mse[idx_test]\
                = model.train(25000)

np.savetxt('final_ans_val_MAE.txt', final_ans_val_mae)
np.savetxt('final_ans_val_MSE.txt', final_ans_val_mse)
np.savetxt('final_ans_test_MAE.txt', final_ans_test_mae)
np.savetxt('final_ans_test_MSE.txt', final_ans_test_mse)
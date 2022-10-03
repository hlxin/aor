from __future__ import print_function, division

import numpy as np

from ase.db import connect

from tinnet.regression.regression import Regression

lr = 0.006326505613656666
lamb = 0.1
atom_fea_len = 139
n_conv = 3
h_fea_len = 253
n_h = 1
Esp = -3.343

db = connect('train.db')

images = np.array([r.toatoms()
                   for r in db.select(purpose='adsorption_energy',
                                      site='bridge')])

ae = np.array([r['data']['ae']
               for r in db.select(purpose='adsorption_energy',
                                  site='bridge')])

d_cen_1 = np.array([r['data']['d_cen_1']
                    for r in db.select(purpose='adsorption_energy',
                                       site='bridge')])

d_cen_2 = np.array([r['data']['d_cen_2']
                    for r in db.select(purpose='adsorption_energy',
                                       site='bridge')])

vad2 = np.array([r['data']['vad2']
                 for r in db.select(purpose='adsorption_energy',
                                    site='bridge')])

width_1 = np.array([r['data']['width_1']
                    for r in db.select(purpose='adsorption_energy',
                                       site='bridge')])

width_2 = np.array([r['data']['width_2']
                    for r in db.select(purpose='adsorption_energy',
                                       site='bridge')])

dos_ads_1 = np.array([r['data']['dos_ads_1']
                      for r in db.select(purpose='adsorption_energy',
                                         site='bridge')])

dos_ads_2 = np.array([r['data']['dos_ads_2']
                      for r in db.select(purpose='adsorption_energy',
                                         site='bridge')])

dos_ads_3 = np.array([r['data']['dos_ads_3']
                      for r in db.select(purpose='adsorption_energy',
                                         site='bridge')])

final_ans_val_mae = np.zeros(10)
final_ans_val_mse = np.zeros(10)
final_ans_test_mae = np.zeros(10)
final_ans_test_mse = np.zeros(10)

for idx_test in range(0,10):
    for idx_validation in range(0,10):
        
        model = Regression(images,
                           ae,
                           task='train',
                           data_format='nested',
                           phys_model='newns_anderson_semi', # for training
                           optim_algorithm='AdamW', # for training
                           batch_size=64, # for training
                           weight_decay=0.0001, # for training
                           idx_validation=idx_validation, # for training
                           idx_test=idx_test, # for training
                           lr=lr, # for architecture
                           atom_fea_len=atom_fea_len, # for architecture
                           n_conv=n_conv, # for architecture
                           h_fea_len=h_fea_len, # for architecture
                           n_h=n_h, # for architecture
                           Esp=Esp, # for TinNet
                           lamb=lamb, # for TinNet
                           d_cen_1=d_cen_1, # for TinNet
                           d_cen_2=d_cen_2, # for TinNet
                           width_1=width_1, # for TinNet
                           width_2=width_2, # for TinNet
                           vad2=vad2, # for TinNet
                           dos_ads_1=dos_ads_1, # for TinNet
                           dos_ads_2=dos_ads_2, # for TinNet
                           dos_ads_3=dos_ads_3, # for TinNet
                           emax=15, # for TinNet
                           emin=-15, # for TinNet
                           num_datapoints=3001 # for TinNet
                           )
        
        final_ans_val_mae[idx_test], \
        final_ans_val_mse[idx_test], \
        final_ans_test_mae[idx_test], \
        final_ans_test_mse[idx_test] \
            = model.train(25000)

np.savetxt('final_ans_val_MAE.txt', final_ans_val_mae)
np.savetxt('final_ans_val_MSE.txt', final_ans_val_mse)
np.savetxt('final_ans_test_MAE.txt', final_ans_test_mae)
np.savetxt('final_ans_test_MSE.txt', final_ans_test_mse)

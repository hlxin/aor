from __future__ import print_function, division

import os
import time
import warnings
import random
import csv
from random import sample
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from piml.phys.phys import Chemisorption


class Regression:
    def __init__(self,
                 features,
                 energy,
                 name_images=None,
                 phys_model='gnn',
                 idx_validation=0,
                 idx_test=None,
                 print_freq=1,
                 batch_size=256,
                 num_workers=0,
                 atom_fea_len=64,
                 n_conv=3,
                 h_fea_len=128,
                 n_h=1,
                 optim_algorithm='Adam',
                 lr=0.001,
                 momentum=0.9,
                 weight_decay=0,
                 lr_milestones=[100],
                 resume=None,
                 random_seed=1234,
                 start_epoch=0,
                 **kwargs
                 ):
        
        # Initialize Physical Model
        Chemisorption.__init__(self, phys_model, **kwargs)
        
        atom_fea = np.array([x[0] for x in features])
        nbr_fea = np.array([x[1] for x in features])
        nbr_fea_idx = np.array([x[2] for x in features])
        
        target = energy
        
        if name_images is None:
            name_images = np.arange(len(atom_fea))

        dataset = [((torch.Tensor(atom_fea[i]),
                     torch.Tensor(nbr_fea[i]),
                     torch.LongTensor(nbr_fea_idx[i])),
                    torch.Tensor([target[i]]),
                    name_images[i])
                   for i in range(len(atom_fea))]
        
        best_mse_error = 0.0
        
        cuda = torch.cuda.is_available()
        
        collate_fn = self.collate_pool
        
        train_loader, val_loader, test_loader =\
            self.get_train_val_test_loader(dataset=dataset,
                                           collate_fn=collate_fn,
                                           batch_size=batch_size,
                                           idx_validation=idx_validation,
                                           idx_test=idx_test,
                                           num_workers=num_workers,
                                           pin_memory=cuda,
                                           random_seed=random_seed)
        
        # obtain target value normalizer
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = self.collate_pool(sample_data_list)
        
        # build model
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len,
                                    nbr_fea_len,
                                    atom_fea_len=atom_fea_len,
                                    n_conv=n_conv,
                                    h_fea_len=h_fea_len,
                                    n_h=n_h,
                                    model_num_input=self.model_num_input)
        
        if cuda:
            model.cuda()
        
        # define loss function
        criterion = nn.NLLLoss()
        
        # define optimizer
        if optim_algorithm == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
        elif optim_algorithm == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr,
                                   weight_decay=weight_decay)
        elif optim_algorithm == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr,
                                    weight_decay=weight_decay)
        else:
            raise NameError('Only SGD, Adam or AdamW is allowed')
        
        # optionally resume from a checkpoint
        if resume:
            if os.path.isfile(resume):
                print('=> loading checkpoint "{}"'.format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_mse_error = checkpoint['best_mse_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint "{}" (epoch {})'
                      .format(resume, checkpoint['epoch']))
            else:
                print('=> no checkpoint found at "{}"'.format(resume))
        
        scheduler = MultiStepLR(optimizer,
                                milestones=lr_milestones,
                                gamma=0.1)
        
        # Initialize the class
        self.lr = lr
        self.cuda = cuda
        self.phys_model = phys_model
        self.print_freq = print_freq
        self.start_epoch = start_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_mse_error = best_mse_error
        self.best_counter = 0
        self.idx_validation = idx_validation
        self.idx_test = idx_test
    
    def train(self, epochs=10, **kwargs):
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            self.epoch = epoch
            train_mse, train_mae = self.train_model(**kwargs)
            
            # evaluate on validation set
            val_mse, val_mae = self.val_model(**kwargs)
            
            if val_mse != val_mse:
                print('Exit due to NaN')
                return 1e10, 1e10, 1e10, 1e10
            
            self.scheduler.step()
            
            # remember the best mse_eror and save checkpoint
            is_best = val_mse < self.best_mse_error
            self.best_mse_error = min(val_mse, self.best_mse_error)
            
            self.save_checkpoint({'epoch': epoch + 1,
                                  'state_dict': self.model.state_dict(),
                                  'best_mse_error': self.best_mse_error,
                                  'optimizer': self.optimizer.state_dict(),
                                  }, is_best, **kwargs)

            if self.best_counter >= 1000:
                print('Exit due to converged')
                filename = 'model_best_train_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.pth.tar'
                torch.save(self.best_state, filename)
                return self.best_val_mae, self.best_val_mse,\
                    self.best_test_mae, self.best_test_mse
        
        return 1e10, 1e10, 1e10, 1e10

    def check_loss(self, **kwargs):
        # test best model
        best_checkpoint = torch.load('model_best_train_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.pth.tar')
        
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        train_mse, train_mae = self.eval_train_model(**kwargs)
        val_mse, val_mae = self.eval_val_model(**kwargs)
        test_mse, test_mae = self.eval_test_model(**kwargs)
        return train_mae, train_mse, val_mae, val_mse, test_mae, test_mse
    
    def eval_train_model(self, **kwargs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        auc_scores = AverageMeter()
        
        # switch to evaluate mode
        self.model.eval()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.train_loader):
            with torch.no_grad():
                if self.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True)
                                  for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
                
                target = target.view(-1).long()
                
                if self.cuda:
                    target_var = Variable(target.cuda(non_blocking=True))
                else:
                    target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gnn':
                output, parm = Chemisorption.gnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    model='model',
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            loss = self.criterion(output, target_var)
            
            # measure accuracy and record loss
            accuracy, auc_score = \
                self.class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    batch_time=batch_time,
                    loss=losses, accu=accuracies,
                    auc=auc_scores))
        
        np.savetxt('parm_train_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.txt', parm.detach().cpu().numpy())
        
        output = torch.exp(output.data.cpu())
        output = output[:, 1]
        
        with open('train_results_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(batch_cif_ids, target, output):
                writer.writerow((cif_id, target.detach().cpu().numpy(), pred.detach().cpu().numpy()))
        
        return losses.avg, auc_scores.avg
    
    def eval_val_model(self, **kwargs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        auc_scores = AverageMeter()
                
        # switch to evaluate mode
        self.model.eval()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.val_loader):
            with torch.no_grad():
                if self.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True)
                                  for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
                
                target = target.view(-1).long()
                
                if self.cuda:
                    target_var = Variable(target.cuda(non_blocking=True))
                else:
                    target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gnn':
                output, parm = Chemisorption.gnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    model='model',
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            loss = self.criterion(output, target_var)
            
            # measure accuracy and record loss
            accuracy, auc_score = \
                self.class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(self.val_loader), batch_time=batch_time,
                    loss=losses, accu=accuracies, auc=auc_scores))
        
        np.savetxt('parm_validation_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.txt', parm.detach().cpu().numpy())
        
        output = torch.exp(output.data.cpu())
        output = output[:, 1]
        
        with open('validation_results_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(batch_cif_ids, target, output):
                writer.writerow((cif_id, target.detach().cpu().numpy(), pred.detach().cpu().numpy()))
        
        return losses.avg, auc_scores.avg
    
    def eval_test_model(self, **kwargs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        auc_scores = AverageMeter()
        
        # switch to evaluate mode
        self.model.eval()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.test_loader):
            with torch.no_grad():
                if self.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True)
                                  for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
                
                target = target.view(-1).long()
                
                if self.cuda:
                    target_var = Variable(target.cuda(non_blocking=True))
                else:
                    target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gnn':
                output, parm = Chemisorption.gnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    model='model',
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            loss = self.criterion(output, target_var)
            
            # measure accuracy and record loss
            accuracy, auc_score = \
                self.class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(self.test_loader), batch_time=batch_time,
                    loss=losses, accu=accuracies, auc=auc_scores))
        
        np.savetxt('parm_test_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.txt', parm.detach().cpu().numpy())
        
        output = torch.exp(output.data.cpu())
        output = output[:, 1]
        
        with open('test_results_idx_val_' + str(self.idx_validation) + '_idx_test_' + str(self.idx_test) + '.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(batch_cif_ids, target, output):
                writer.writerow((cif_id, target.detach().cpu().numpy(), pred.detach().cpu().numpy()))
        
        return losses.avg, auc_scores.avg
    
    def train_model(self, **kwargs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        
        # switch to train mode
        self.model.train()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
    
            if self.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True)
                              for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
            
            if self.cuda:
                target_var = Variable(target.cuda(non_blocking=True))
            else:
                target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gnn':
                output, parm = Chemisorption.gnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    model='dft',
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            loss = self.criterion(output, target_var)*output.shape[-1]
            
            # measure accuracy and record loss
            mae_error = self.mae(output.data.cpu(),target)*output.shape[-1]
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            
            # compute gradient and do SGD/Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'
                      .format(self.epoch, i, len(self.train_loader),
                              batch_time=batch_time,
                              data_time=data_time,
                              loss=losses,
                              mae_errors=mae_errors))
        
        return losses.avg, auc_scores.avg
    
    def val_model(self, **kwargs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        
        # switch to evaluate mode
        self.model.eval()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.val_loader):
            with torch.no_grad():
                if self.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True)
                                  for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
                
                if self.cuda:
                    target_var = Variable(target.cuda(non_blocking=True))
                else:
                    target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gnn':
                output, parm = Chemisorption.gnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    model='model',
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            loss = self.criterion(output, target_var)*output.shape[-1]
            
            # measure accuracy and record loss
            mae_error = self.mae(output.data.cpu(),target)*output.shape[-1]
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % self.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'
                      .format(i, len(self.val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              mae_errors=mae_errors))
        
        return losses.avg, auc_scores.avg
    
    def test_model(self, **kwargs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        
        # switch to evaluate mode
        self.model.eval()
        
        end = time.time()
        
        for i, (input, target, batch_cif_ids) in enumerate(self.test_loader):
            with torch.no_grad():
                if self.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True)
                                  for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
                
                if self.cuda:
                    target_var = Variable(target.cuda(non_blocking=True))
                else:
                    target_var = Variable(target)
            
            # compute output
            cnn_output = self.model(*input_var)
            
            if self.phys_model =='gnn':
                output, parm = Chemisorption.gnn(self,
                                                  cnn_output,
                                                  **kwargs)
            
            if self.phys_model =='newns_anderson_semi':
                output, parm = Chemisorption.newns_anderson_semi(
                    self,
                    cnn_output,
                    model='model',
                    **dict(**kwargs, batch_cif_ids=batch_cif_ids))
            
            loss = self.criterion(output, target_var)*output.shape[-1]
            
            # measure accuracy and record loss
            mae_error = self.mae(output.data.cpu(),target)*output.shape[-1]
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'
                      .format(i, len(self.test_loader),
                              batch_time=batch_time,
                              loss=losses,
                              mae_errors=mae_errors))
        
        return losses.avg, auc_scores.avg
    
    def mae(self, prediction, target):
        '''
        Computes the mean absolute error between prediction and target
    
        Parameters
        ----------
    
        prediction: torch.Tensor (N, 1)
        target: torch.Tensor (N, 1)
        '''
        return torch.mean(torch.abs(target - prediction))
    
    def save_checkpoint(self, state, is_best, **kwargs):
        self.best_counter += 1
        if is_best:
            self.best_state = deepcopy(state)
            self.best_val_mse, self.best_val_mae = self.val_model(**kwargs)
            self.best_test_mse, self.best_test_mae = self.test_model(**kwargs)
            self.best_counter = 0
    
    def get_train_val_test_loader(self,
                                  dataset,
                                  idx_validation=0,
                                  idx_test=None,
                                  collate_fn=default_collate,
                                  batch_size=256,
                                  num_workers=0,
                                  pin_memory=False,
                                  random_seed=None):
        '''
        Utility function for dividing a dataset to train, val, test datasets.
    
        The dataset needs to be shuffled before using the function
    
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
          The full dataset to be divided.
        batch_size: int
        train_ratio: float
        val_ratio: float
        test_ratio: float
        num_workers: int
        pin_memory: bool
    
        Returns
        -------
        train_loader: torch.utils.data.DataLoader
          DataLoader that random samples the training data.
        val_loader: torch.utils.data.DataLoader
          DataLoader that random samples the validation data.
        test_loader: torch.utils.data.DataLoader
          DataLoader that random samples the test data.
        '''
        
        indices = np.arange(len(dataset))
        
        kfold_test = deepcopy(indices)
        kfold_val = deepcopy(indices)
        kfold_train = deepcopy(indices)
        
        val_sampler = SubsetRandomSampler(deepcopy(kfold_val))
        test_sampler = SubsetRandomSampler(deepcopy(kfold_test))
        train_sampler = SubsetRandomSampler(deepcopy(kfold_train))
        
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  pin_memory=pin_memory)
        
        val_loader = DataLoader(dataset, batch_size=1024,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                collate_fn=collate_fn,
                                pin_memory=pin_memory)
        
        test_loader = DataLoader(dataset, batch_size=1024,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)
        
        return train_loader, val_loader, test_loader

    def collate_pool(self, dataset_list):
        '''
        Collate a list of data and return a batch for predicting crystal
        properties.
    
        Parameters
        ----------
    
        dataset_list: list of tuples for each data point.
          (atom_fea, nbr_fea, nbr_fea_idx, target)
    
          atom_fea: torch.Tensor shape (n_i, atom_fea_len)
          nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
          nbr_fea_idx: torch.LongTensor shape (n_i, M)
          target: torch.Tensor shape (1, )
          cif_id: str or int
    
        Returns
        -------
        N = sum(n_i); N0 = sum(i)
    
        batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type
        batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        batch_nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        target: torch.Tensor shape (N, 1)
          Target value for prediction
        batch_cif_ids: list
        '''
        batch_atom_fea = []
        batch_nbr_fea = []
        batch_nbr_fea_idx = []
        crystal_atom_idx = []
        batch_target = []
        batch_cif_ids = []
        base_idx = 0
        
        for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
                in enumerate(dataset_list):
            n_i = atom_fea.shape[0]  # number of atoms for this crystal
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
            new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
            crystal_atom_idx.append(new_idx)
            batch_target.append(target)
            batch_cif_ids.append(cif_id)
            base_idx += n_i
        return (torch.cat(batch_atom_fea, dim=0),
                torch.cat(batch_nbr_fea, dim=0),
                torch.cat(batch_nbr_fea_idx, dim=0),
                crystal_atom_idx),\
            torch.stack(batch_target, dim=0),\
            batch_cif_ids
    
    def class_eval(self, prediction, target):
        prediction = np.exp(prediction.numpy())
        target = target.numpy()
        pred_label = np.argmax(prediction, axis=1)
        target_label = np.squeeze(target)
        if prediction.shape[1] == 2:
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
            accuracy = metrics.accuracy_score(target_label, pred_label)
        else:
            raise NotImplementedError
        return accuracy, auc_score


class AverageMeter:
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConvLayer(nn.Module):
    '''
    Convolutional operation on graphs
    '''
    def __init__(self, atom_fea_len, nbr_fea_len):
        '''
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        '''
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        '''
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        '''
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    '''
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    '''
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 model_num_input=1):
        '''
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        '''
        super(CrystalGraphConvNet, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len, model_num_input)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout()
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        '''
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        '''
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        out = self.fc_out(crys_fea)
        out = self.logsoftmax(out)
        
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        '''
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        '''
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

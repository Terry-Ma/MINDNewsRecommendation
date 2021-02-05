import torch
import logging
import os
import numpy as np

from torch import nn
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from NRMS.module import NRMS
from NRMS.data_loader import NRMSDataLoader
from optimizer import *

name2dataloader = {
    'NRMS': NRMSDataLoader
    }

name2module = {
    'NRMS': NRMS
    }

name2optimizer = {
    'NRMS': NNOptimizer
    }

logger = logging.getLogger()

class NNModel:
    def __init__(self, config):
        self.config = config
        self.init_data()
        self.init_device()
        self.init_model()
        self.init_loss()
        self.init_optimizer()

    def init_data(self):
        self.data_loader = name2dataloader[self.config['model']['model_name']](self.config)
        self.config['model']['vocab_size'] = self.data_loader.get_vocab_size()

    def init_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('use device {}'.format(self.device))

    def init_model(self):
        self.model = name2module[self.config['model']['model_name']](self.config).to(self.device)
        logger.info('init model \n{}'.format(self.model))
        # multi-gpu
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info('will train on {} gpus'.format(torch.cuda.device_count()))

    def init_loss(self):
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        logger.info('init loss \n{}'.format(self.loss))

    def init_optimizer(self):
        self.optimizer = name2optimizer[self.config['model']['model_name']](self.config, self.model)
        logger.info('init optimizer\n {}'.format(self.optimizer))

    def train(self):
        # init metric
        self.cur_epochs = 1
        self.cur_train_steps = 1
        self.max_val_auc = 0
        self.best_step = 1
        # load checkpoint
        if self.config['model']['load_checkpoint_path'] != '':
            self.load_checkpoint()
        # init writer
        tb_writer = SummaryWriter(log_dir=self.config['train']['tb_path'])
        # total steps
        total_steps = self.cur_train_steps + self.config['train']['train_steps']
        logger.info('start training')
        while self.cur_train_steps <= total_steps:
            check_train_steps = 0
            check_train_loss = 0
            check_train_ys = []
            check_train_pred_ys = []
            # shuffle every epoch
            for batch in self.data_loader.train_batch_iter():
                batch_y = batch.pop('labels')
                check_train_ys.append(np.array(batch_y))
                # train
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                batch_y = batch_y.to(self.device)
                batch_pred_y = self.model(**batch)
                batch_pred_y = torch.cat((torch.zeros(batch_pred_y.shape[0], 1).to(self.device), \
                    batch_pred_y.unsqueeze(1)), dim=1)
                train_loss = self.loss(batch_pred_y, batch_y)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                # check train steps & loss & pred_y
                check_train_steps += 1
                check_train_loss += train_loss.item()
                check_train_pred_ys.append(np.array(batch_pred_y[:, 1].detach().to('cpu')))
                # check & val step
                if self.cur_train_steps > 0 and self.cur_train_steps % self.config['train']['steps_per_check'] == 0:
                    self.model.eval()    # dropout...
                    # train auc
                    check_train_y = np.concatenate(check_train_ys)
                    check_train_pred_y = np.concatenate(check_train_pred_ys)
                    check_train_auc = roc_auc_score(check_train_y, check_train_pred_y)
                    # val auc
                    with torch.no_grad():
                        self.data_loader.generate_embedding(self.model)
                        check_val_loss = 0
                        check_val_steps = 0
                        check_val_ys = []
                        check_val_pred_ys = []
                        for batch_pred_y, batch_y in self.data_loader.val_inference():
                            check_val_ys.append(np.array(batch_y))
                            # check val steps & loss & pred_y
                            batch_pred_y = torch.cat((torch.zeros(batch_pred_y.shape[0], 1), \
                                batch_pred_y.unsqueeze(1)), dim=1)
                            check_val_loss += self.loss(batch_pred_y, batch_y).item()
                            check_val_steps += 1
                            check_val_pred_ys.append(np.array(batch_pred_y[:, 1].to('cpu')))
                    check_val_y = np.concatenate(check_val_ys)
                    check_val_pred_y = np.concatenate(check_val_pred_ys)
                    check_val_auc = roc_auc_score(check_val_y, check_val_pred_y)
                    # max auc model
                    if check_val_auc > self.max_val_auc:
                        self.max_val_auc = check_val_auc
                        self.best_step = self.cur_train_steps
                        cpt_path = '{}/best_model.cpt'.format(self.config['train']['checkpoint_path'])
                        self.save_checkpoint(cpt_path)
                    # log
                    logger.info('epoch {0}, steps {1}, train_loss {2:.4f}, val_loss {3:.4f}, train_auc {4:.4f}, val_auc {5:.4f}, max_val_auc {6:.4f}, best_step {7}, lr {8}'.format(
                        self.cur_epochs,
                        self.cur_train_steps,
                        check_train_loss / check_train_steps,
                        check_val_loss / check_val_steps,
                        check_train_auc,
                        check_val_auc,
                        self.max_val_auc,
                        self.best_step,
                        self.optimizer.lr
                        ))
                    tb_writer.add_scalar('Loss/train_loss', check_train_loss / check_train_steps, self.cur_train_steps)
                    tb_writer.add_scalar('Loss/val_loss', check_val_loss / check_val_steps, self.cur_train_steps)
                    tb_writer.add_scalar('AUC/train_auc', check_train_auc, self.cur_train_steps)
                    tb_writer.add_scalar('AUC/val_auc', check_val_auc, self.cur_train_steps)
                    tb_writer.add_scalar('lr', self.optimizer.lr, self.cur_train_steps)
                    # init
                    check_train_steps = 0
                    check_train_loss = 0
                    check_train_y = np.array([])
                    check_train_pred_y = np.array([])
                    self.model.train()    # dropout...
                # checkpoint
                if self.cur_train_steps > 0 and self.cur_train_steps % self.config['train']['steps_per_checkpoint'] == 0:
                    cpt_path = '{}/checkpoint_steps_{}.cpt'.format(self.config['train']['checkpoint_path'], self.cur_train_steps)
                    self.save_checkpoint(cpt_path)
                self.cur_train_steps += 1
                if self.cur_train_steps > total_steps:
                    break
            if self.cur_train_steps <= total_steps:
                self.cur_epochs += 1
        logger.info('training complete, training epochs {0}, steps {1}, max_val_auc {2:.4f}, best_step {3}'.\
            format(self.cur_epochs, self.config['train']['train_steps'], self.max_val_auc, self.best_step))
        tb_writer.close()
        # generate submit
        self.generate_submit()

    def save_checkpoint(self, cpt_path):
        cpt_dict = {  # DataParallel's state_dict is in module
            'model': self.model.module.state_dict() if torch.cuda.device_count() > 1 \
                else self.model.state_dict(),
            'cur_train_steps': self.cur_train_steps,
            'cur_epochs': self.cur_epochs,
            'max_val_auc': self.max_val_auc,
            'best_step': self.best_step
            }
        torch.save(cpt_dict, cpt_path)
        logger.info('save checkpoints {}'.format(cpt_path))

    def load_checkpoint(self):
        logger.info('will load checkpoint')
        cpt_path = self.config['model']['load_checkpoint_path']
        if os.path.exists(cpt_path):
            # load checkpoint
            cpt_dict = torch.load(cpt_path)
            self.model.load_state_dict(cpt_dict['model'])
            self.cur_train_steps = cpt_dict['cur_train_steps'] + 1
            self.cur_epochs = cpt_dict['cur_epochs'] + 1
            self.max_val_auc = cpt_dict['max_val_auc']
            self.best_step = cpt_dict['best_step']
            logger.info('load checkpoint from {}'.format(cpt_path))
        else:
            logger.error('checkpoint path {} not exists'.format(cpt_path))
            raise Exception('checkpoint path {} not exists'.format(cpt_path))

    def generate_submit(self):
        logger.info('start generating submit')
        # load best model
        cpt_path = '{}/best_model.cpt'.format(self.config['train']['checkpoint_path'])
        if os.path.exists(cpt_path):
            logger.info('load best model')
            cpt_dict = torch.load(cpt_path)
            self.model.load_state_dict(cpt_dict['model'])
        # generate embedding
        logger.info('start generating embedding')
        self.model.eval()
        with torch.no_grad():
            self.data_loader.generate_embedding(self.model)
        test_iid2num = self.data_loader.get_test_iid2num()
        cur_iid = 1
        cur_impr_scores = []
        logger.info('start writing submit file')
        with open(self.config['train']['submit_path'], encoding='utf-8', mode='w') as f:
            for batch_scores in self.data_loader.test_inference():
                batch_scores_idx = 0
                while batch_scores_idx + test_iid2num[cur_iid] - len(cur_impr_scores) <= batch_scores.shape[0]:
                    # write
                    cur_impr_scores += list(batch_scores[
                        batch_scores_idx:batch_scores_idx + test_iid2num[cur_iid] - len(cur_impr_scores)].numpy())
                    cur_impr_scores = [[cur_impr_scores[i], i] for i in range(len(cur_impr_scores))]
                    cur_impr_scores = sorted(cur_impr_scores, key=lambda x: x[0], reverse=True)
                    cur_impr_ranks = [i[1] for i in cur_impr_scores]
                    f.write('{} {}\n'.format(cur_iid, cur_impr_ranks))
                    # state update
                    batch_scores_idx += test_iid2num[cur_iid] - len(cur_impr_scores)
                    cur_impr_scores = []
                    cur_iid += 1        
                cur_impr_scores += list(batch_scores[batch_scores_idx:].numpy())
        logging.info('successfully generating submit {}'.format(self.config['train']['submit_path']))
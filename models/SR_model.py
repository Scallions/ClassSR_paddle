import logging
from collections import OrderedDict
import paddle 
import paddle.nn as nn
from paddle.distributed import fleet
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss,class_loss_3class,average_loss_3class
from models.archs import arch_util
import cv2
import numpy as np
from utils import util
from data import util as ut
import os.path as osp
import os


logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)



        if opt['dist']:
            self.rank = paddle.distributed.ParallelEnv().rank
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt)#.to(self.device)
        # print network
        self.print_network()
        self.load()


        if opt['dist']:
            self.netG = fleet.distributed_model(self.netG)

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss()#.to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss()#.to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss()#.to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0.0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                # if v.requires_grad:
                if not v.stop_gradient:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            # schedulers
            # if train_opt['lr_scheme'] == 'MultiStepLR':
            #     for optimizer in self.optimizers:
            #         self.schedulers.append(
            #             lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['T_period'],
            #                                              restarts=train_opt['restarts'],
            #                                              weights=train_opt['restart_weights'],
            #                                              gamma=train_opt['lr_gamma'],
            #                                              clear_state=train_opt['clear_state']))
            # elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            #     for optimizer in self.optimizers:
            #         self.schedulers.append(
            #             # lr_scheduler.CosineAnnealingLR_Restart(
            #                 # optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
            #             lr_scheduler.CosineAnnealingDecay(
            #                 train_opt['lr_G'], train_opt['T_period'], eta_min=train_opt['eta_min'],
            #                 restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            # else:
            #     raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            
            if train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingDecay(train_opt['lr_G'],
                        train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
                    #optimizer._learning_rate = self.schedulers[-1]
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.optimizer_G = paddle.optimizer.Adam(learning_rate=self.schedulers[0], parameters=optim_params,
                                                     weight_decay=wd_G,
                                                     beta1=train_opt['beta1'], beta2=train_opt['beta2'])
            self.optimizers.append(self.optimizer_G)
            if opt['dist']:
                self.optimizer_G = fleet.distributed_optimizer(self.optimizer_G)

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ']#.to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT']#.to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.clear_grad()
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with paddle.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].astype('float')#().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].astype('float')#().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].astype('float')#().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        # if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
        #     net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
        #                                      self.netG.module.__class__.__name__)
        # else:
        net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n.item()))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

